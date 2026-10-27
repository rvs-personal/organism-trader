"""Local-only Ollama tool loop. No API key, cloud fallback or per-token billing."""
import json
from urllib.parse import urlparse
import httpx
from .base import Backend, EpisodeResult, Usage


class OllamaBackend(Backend):
    name='ollama'
    supports_json_schema=True

    def __init__(self,base_url='http://127.0.0.1:11434',timeout=180,num_ctx=16384,num_predict=1800,temperature=.2,thinking=False,tool_mode='native'):
        if urlparse(base_url).hostname not in {'localhost','127.0.0.1','::1'}:
            raise ValueError('Ollama backend is restricted to a local server')
        self.base_url=base_url.rstrip('/');self.client=httpx.Client(timeout=timeout)
        self.options=dict(num_ctx=num_ctx,num_predict=num_predict,temperature=temperature,seed=42)
        self.thinking = thinking
        if tool_mode not in ('native','json'): raise ValueError('Unknown local tool mode')
        self.tool_mode = tool_mode

    def run_episode(self,system,user,tools,tool_specs,model_id,max_tool_turns=4,output_schema=None):
        if 'cloud' in model_id.lower():raise ValueError('Cloud model tags are not allowed in the local backend')
        messages=[dict(role='system',content=system),dict(role='user',content=user)]
        payload=[dict(type='function',function=dict(name=n,description=s['description'],parameters=s['input_schema'])) for n,s in tool_specs.items() if n in tools]
        if self.tool_mode == 'json' and payload and max_tool_turns:
            messages[0]['content'] += '\nResearch tools are Python functions, invoked through JSON. Return {"tool_requests":[{"name":"tool_name","arguments":{...}}]} to request information, or return the final requested schema directly. Tool requests do not place orders. Available tools:\n' + json.dumps(payload)
        usage=Usage();trace=[];stats=[]
        for turn in range(max_tool_turns+1):
            from ..host_health import snapshot, should_pause
            if should_pause(snapshot()):
                raise RuntimeError('Local inference resource pressure: retry when laptop memory/thermal pressure recovers')
            body=dict(model=model_id,messages=messages,stream=False,think=self.thinking,keep_alive='30m',options=self.options)
            if payload and turn<max_tool_turns and self.tool_mode == 'native':body['tools']=payload
            # Final content is constrained to the requested schema; native tool calls stay structured.
            if output_schema:body['format']=output_schema
            if payload and turn<max_tool_turns and self.tool_mode == 'json':
                choices=[dict(type='object',properties={'name':{'const':x['function']['name']},'arguments':x['function']['parameters']},required=['name','arguments'],additionalProperties=False) for x in payload]
                request_schema=dict(type='object',properties={'tool_requests':dict(type='array',items={'anyOf':choices},minItems=1,maxItems=16)},required=['tool_requests'],additionalProperties=False)
                body['format']={'$defs':(output_schema or {}).get('$defs',{}),'anyOf':[request_schema,{k:v for k,v in (output_schema or {'type':'object'}).items() if k!='$defs'}]}
            response=self.client.post(self.base_url+'/api/chat',json=body);response.raise_for_status();r=response.json()
            if r.get('error'):raise RuntimeError(r['error'])
            if r.get('done_reason')=='length':raise RuntimeError('Local model output token limit reached; refusing truncated decisions')
            if r.get('prompt_eval_count', 0) >= self.options['num_ctx'] - self.options['num_predict']:
                raise RuntimeError('Local prompt exceeds reserved context budget; refusing possibly truncated evidence')
            msg=r.get('message',{});messages.append(msg)
            usage.add(Usage(input_tokens=r.get('prompt_eval_count',0),output_tokens=r.get('eval_count',0),cost_inr=0.0))
            stats.append({k:r.get(k) for k in ['total_duration','load_duration','prompt_eval_count','prompt_eval_duration','eval_count','eval_duration']})
            calls=msg.get('tool_calls') or []
            if self.tool_mode == 'json':
                parsed=json.loads(msg.get('content') or '{}')
                calls=[dict(function=dict(name=c['name'],arguments=c['arguments'])) for c in parsed.get('tool_requests',[])]
            if not calls:return EpisodeResult(msg.get('content',''),usage,trace,model_id,dict(local_stats=stats))
            if turn>=max_tool_turns:raise RuntimeError('Local model exceeded the tool-turn budget')
            if len(calls)>16:raise RuntimeError('Local model requested too many tools in one turn')
            for call in calls:
                f=call['function'];name=f['name'];args=f.get('arguments',{})
                if isinstance(args,str):args=json.loads(args)
                if not isinstance(args,dict):raise ValueError('Tool arguments must be an object')
                try:
                    result=tools[name](args) if name in tools else {'error':'Unknown or unavailable tool'}
                except (KeyError,TypeError,ValueError) as error:
                    result={'error':str(error)}
                trace.append(dict(tool=name,args=args))
                if self.tool_mode == 'json':
                    messages.append(dict(role='user',content=f'Python tool result for {name}:\n'+json.dumps(result)+'\nContinue research or return the final schema.'))
                else:
                    messages.append(dict(role='tool',tool_name=name,content=json.dumps(result)))
        raise RuntimeError('Local episode did not produce a decision')
