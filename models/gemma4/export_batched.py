"""Experimental functional-KV Gemma4 batch asset. All work is local to the mini.

Separate artifact/manifest: the shipping static engine must not load this ABI.
Initial geometry is deliberately small; deployment requires full-model parity.
"""
import argparse,gc,hashlib,importlib.util,json,logging,os,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from coreai.authoring.types import AllocationType,HardwareConstraints
from coreai_torch import TorchConverter
from coreai_models.models.ios.gemma4_text import Gemma4ForCausalLMForiOS
from coreai_models.models.ios.gemma4_batched import BatchedReadonlyGemma4,FlatKVBatchedReadonlyGemma4
from coreai_models.export.mlir_ops import register_custom_torch_lowering
spec=importlib.util.spec_from_file_location('shipping_gemma_export',Path(__file__).with_name('export.py'))
shipping=importlib.util.module_from_spec(spec);spec.loader.exec_module(shipping)
log=logging.getLogger('batch-export')

def load_mapped(source,cache,context):
    cfg=Gemma4ForCausalLMForiOS._get_reauthored_config(AutoConfig.from_pretrained(str(source)),context)
    model=Gemma4ForCausalLMForiOS(cfg,model_device='meta').to(dtype=torch.float16)
    cache.mkdir(parents=True,exist_ok=True);sd={}
    with safe_open(source/'model.safetensors',framework='pt') as f:
        keys=[k for k in f.keys() if k.startswith('model.language_model.')]
        for index,key in enumerate(keys):
            view=f.get_slice(key);shape=view.get_shape();count=int(np.prod(shape))
            if count>=8_000_000:
                filename=cache/(hashlib.sha256(key.encode()).hexdigest()+'.npy')
                if not filename.exists():
                    temporary=filename.with_suffix('.partial.npy')
                    dest=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float16,shape=tuple(shape))
                    rows=max(1,8_000_000//max(1,int(np.prod(shape[1:]))))
                    for start in range(0,shape[0],rows):dest[start:start+rows]=view[start:start+rows].half().numpy()
                    dest.flush();del dest;os.replace(temporary,filename)
                mapped=np.load(filename,mmap_mode='r+')
                if mapped.shape!=tuple(shape) or mapped.dtype!=np.float16:raise ValueError(f'Invalid mmap: {key}')
                sd[key]=torch.from_numpy(mapped)
            else:sd[key]=f.get_tensor(key).half()
            if index%40==0:log.info('Loaded %d/%d tensors',index,len(keys))
    model._mutate_state_dict(sd)
    model.load_state_dict(sd,strict=True,assign=True)
    del sd;gc.collect()
    return model.eval(),cfg

def main():
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--context',type=int,default=2048);ap.add_argument('--no-compression',action='store_true');ap.add_argument('--kv-layout', choices=['auto','interleave8'], default='auto');ap.add_argument('--shapes',default='1:32,1:1,2:1');a=ap.parse_args()
    if a.output.exists():raise SystemExit('Use a new output directory; never overwrite an active asset.')
    a.output.mkdir(parents=True)
    shapes=[tuple(map(int,s.split(':'))) for s in a.shapes.split(',')]
    if any(b not in (1,2,4) or q not in (1,8,32,64) for b,q in shapes):raise ValueError('unsupported geometry')
    model,cfg=load_mapped(a.source,a.output.parent/'mapped-fp16',a.context)
    log.info('Writing PLE sidecar on mini')
    ple=model.dump_ple_embedding(str(a.output),'gemma4-batched');del model._ple_weight;gc.collect()
    save_file({'embedding_table':model.load_embeddings.embedding_table.detach().contiguous(),
               'scale':model.gather_embeddings.scale.detach().contiguous(),
               'zero_point':model.gather_embeddings.zero_point.detach().contiguous()},str(a.output/'embeddings.safetensors'))
    if not a.no_compression:
        os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(Path(__file__).resolve().parents[2] / "experiments/ane-batch/torch-extensions"))
        from coreai_opt.deps._kmeans1d.core import cluster
        cluster([0., 1., 8., 9.], 2)  # Build once before spawning workers.
        log.info('Applying shipping 4-bit/8-bit palettization')
        config=shipping._load_compression_config_object(Path(__file__).with_name('4bit_palettized.yaml'),'iOS')
        model=shipping.palettize_pytorch_model(model,shipping._palettization_inputs(model,cfg,a.context),config,num_workers=2)
    decoder=FlatKVBatchedReadonlyGemma4(BatchedReadonlyGemma4(model.extend)).eval()
    names=['transformer_input','rope_cos','rope_sin','causal_mask','sliding_causal_mask','key_cache','value_cache','sliding_key_cache','sliding_value_cache','embedding_table','ple_embeddings']
    outs=['logits','new_global_k','new_global_v','new_sliding_k','new_sliding_v']
    conv=TorchConverter();register_custom_torch_lowering(conv);contracts=[]
    ng=model.extend.model.n_global_storing;ns=model.extend.model.n_sliding_storing;ring=model.SLIDING_RING_SIZE
    for b,q in shapes:
        name=f'decode_b{b}_q{q}_c{a.context}';log.info('Tracing %s',name)
        gm=torch.full((b,a.context+q,1,q),-40000.,dtype=torch.float16);sm=torch.full((b,ring+q,1,q),-40000.,dtype=torch.float16)
        for j in range(q):gm[:,a.context:a.context+j+1,0,j]=0;sm[:,ring:ring+j+1,0,j]=0
        k=torch.zeros(ng*b,cfg.num_key_value_heads*cfg.global_head_dim,1,a.context,dtype=torch.float16)
        sk=torch.zeros(ns*b,cfg.num_key_value_heads*cfg.head_dim,1,ring,dtype=torch.float16)
        args=(torch.zeros(b,q,1,cfg.hidden_size,dtype=torch.float16),torch.ones(b,q,cfg.head_dim+cfg.global_head_dim,dtype=torch.float16),torch.zeros(b,q,cfg.head_dim+cfg.global_head_dim,dtype=torch.float16),gm,sm,k,k.clone(),sk,sk.clone(),model.load_embeddings.embedding_table,torch.zeros(b,q,1,cfg.num_hidden_layers*cfg.hidden_size_per_layer_input,dtype=torch.int8))
        with torch.no_grad():ep=torch.export.export(decoder,args).run_decompositions(shipping._ios_decomp_table())
        conv.add_exported_program(ep,input_names=names,output_names=outs,entrypoint_name=name)
        contracts.append({'name':name,'request_batch':b,'query_width':q,'context_capacity':a.context})
        del ep,args;gc.collect()
    log.info('Converting complete decoder graphs')
    program=conv.to_coreai()
    for contract in contracts:
        q=contract['query_width'];constraints={'embedding_table':HardwareConstraints(AllocationType.IOSurface,interleave=[8,1,1],alignments=[1,1,1,1])}
        for n in (names[5:9]+outs[1:] if a.kv_layout == 'interleave8' else []):
            size=q if n.startswith('new_') else (ring if n.startswith('sliding_') else a.context)
            constraints[n]=HardwareConstraints(AllocationType.IOSurface,interleave=[1,8,1,1],alignments=[1,1,1,8*size,1])
        program.set_hardware_constraints(contract['name'],constraints)
    program.save_asset(a.output/'model.aimodel')
    for name in ('tokenizer.json','tokenizer_config.json','generation_config.json','config.json','chat_template.jinja'):
        shutil.copy2(a.source/name,a.output/name)
    manifest={'abi':'gemma4-readonly-batch-v1','status':'experimental-unvalidated','source_revision':'4adf15748de9e1a0097b6274cee9c15836c41c37','context_capacity':a.context,'kv_layout':a.kv_layout,'sliding_ring':ring,'sliding_window':cfg.sliding_window,'global_storing_layers':ng,'sliding_storing_layers':ns,'functions':contracts,'inputs':names,'outputs':outs,'embedding_file':'embeddings.safetensors','ple_file':Path(ple).name,'compression':'none' if a.no_compression else 'shipping-4bit-palettized','text_config':cfg.to_dict()}
    (a.output/'batch-manifest.json').write_text(json.dumps(manifest,indent=2))
    # Prepack the immutable output embedding once on the export machine. The
    # Swift loader checks the runtime descriptor before using this layout.
    with safe_open(a.output/'embeddings.safetensors',framework='np') as f:
        table=f.get_tensor('embedding_table');v,_,h=table.shape
        table.reshape(v//8,8,h).transpose(0,2,1).copy().tofile(a.output/'embedding_interleaved8.bin')
    (a.output/'tokenizer').mkdir()
    for name in ('tokenizer.json','tokenizer_config.json','generation_config.json','config.json','chat_template.jinja'):
        shutil.copy2(a.output/name,a.output/'tokenizer'/name)
    metadata={'metadata_version':'0.2','kind':'llm','name':'gemma4-e2b-ane-batch-experimental',
        'assets':{'main':'model.aimodel'},'auxiliary_assets':{'per_layer_embeddings':Path(ple).name},
        'language':{'tokenizer':'google/gemma-4-E2B-it','vocab_size':cfg.vocab_size,
            'max_context_length':a.context,'max_batch_size':max(b for b,q in shapes),'embedded_tokenizer':True}}
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    log.info('Exported %s; NOT validated for serving',a.output)
if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s');main()
