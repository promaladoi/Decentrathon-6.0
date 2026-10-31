"""Independent candidate diagnostics, never a training score."""
import importlib,json,time,traceback
from pathlib import Path

def main():
    for name in ('benchmark_dropout','benchmark_packed_integration'):
        began=time.perf_counter()
        report={'name':name,'kind':'candidate_diagnostic_not_training_score'}
        try:
            report['results']=importlib.import_module('solution.'+name).run_tuning()
            report['completed']=True
        except Exception as exc:
            report.update(completed=False,error=f'{type(exc).__name__}: {exc}',
                          traceback=traceback.format_exc())
        report['elapsed_s']=time.perf_counter()-began
        (Path('/kaggle/working')/(name+'.json')).write_text(
            json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'name':name,'completed':report['completed'],
                          'elapsed_s':report['elapsed_s']},indent=2),flush=True)

if __name__=='__main__':main()
