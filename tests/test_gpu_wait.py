import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import task12_suite as suite
from util.gpu_wait import (GpuMemory, GpuWaitPolicy, is_cuda_allocation_failure,
                           query_gpu, reserve_gpu, worker_environment)


BUSY=GpuMemory(0,'GPU-test',24576,1135)
READY=GpuMemory(0,'GPU-test',24576,23000)


class Clock:
    def __init__(self):self.now=0.;self.sleeps=[]
    def monotonic(self):return self.now
    def sleep(self,seconds):self.sleeps.append(seconds);self.now+=seconds


class GpuAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_queries_are_bounded_and_response_is_validated(self):
        result=subprocess.CompletedProcess([],0,'0, GPU-test, 24576, 1135\n','')
        with patch('util.gpu_wait.subprocess.run',return_value=result) as run:
            self.assertEqual(query_gpu('0'),BUSY)
        self.assertEqual(run.call_args.kwargs['timeout'],10)
        for output in ['', '0, GPU-test, N/A, N/A', '0, GPU-test, 10, 11',
                       '0, GPU-test, 10, 1\n1, GPU-other, 10, 1']:
            with patch('util.gpu_wait.subprocess.run',return_value=subprocess.CompletedProcess([],0,output,'')):
                with self.assertRaisesRegex(RuntimeError,'response'):query_gpu('0')

    def test_missing_tool_driver_failure_and_query_timeout_are_not_infinite_waits(self):
        for error in [FileNotFoundError(),subprocess.CalledProcessError(1,['nvidia-smi'],stderr='driver failed'),
                      subprocess.TimeoutExpired(['nvidia-smi'],10)]:
            with patch('util.gpu_wait.subprocess.run',side_effect=error):
                with self.assertRaises(RuntimeError):
                    with reserve_gpu(GpuWaitPolicy(),self.root):self.fail('must not admit')

    def test_busy_gpu_waits_without_launching_until_memory_is_free(self):
        clock=Clock();entered=[]
        with patch('util.gpu_wait.query_gpu',side_effect=[BUSY,BUSY,READY]) as query, \
             patch('util.gpu_wait.time.monotonic',clock.monotonic), \
             patch('util.gpu_wait.time.sleep',clock.sleep), contextlib.redirect_stdout(io.StringIO()) as output:
            with reserve_gpu(GpuWaitPolicy(poll_seconds=30),self.root) as gpu:entered.append(gpu)
        self.assertEqual(entered,[READY]);self.assertEqual(clock.sleeps,[30])
        self.assertEqual(query.call_args.args,('GPU-test',))
        self.assertIn('GPU waiting',output.getvalue());self.assertIn('GPU ready',output.getvalue())

    def test_timeout_and_impossible_memory_requirement_fail_clearly(self):
        clock=Clock()
        with patch('util.gpu_wait.query_gpu',return_value=BUSY), \
             patch('util.gpu_wait.time.monotonic',clock.monotonic), \
             patch('util.gpu_wait.time.sleep',clock.sleep), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(TimeoutError):
                with reserve_gpu(GpuWaitPolicy(poll_seconds=40,timeout_seconds=60),self.root):self.fail()
            self.assertEqual(clock.sleeps,[40,20])
            with self.assertRaisesRegex(ValueError,'exceeds'):
                with reserve_gpu(GpuWaitPolicy(min_free_mib=99999),self.root):self.fail()

    def test_cooperative_lock_is_held_through_worker_and_released_on_exception(self):
        import fcntl
        with patch('util.gpu_wait.query_gpu',return_value=READY),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'worker failed'):
                with reserve_gpu(GpuWaitPolicy(),self.root):
                    with (self.root/'GPU-test.lock').open('a') as other:
                        with self.assertRaises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    raise RuntimeError('worker failed')
            with reserve_gpu(GpuWaitPolicy(),self.root) as gpu:self.assertEqual(gpu,READY)

    def test_other_launcher_causes_wait_and_ctrl_c_remains_available(self):
        with patch('util.gpu_wait.query_gpu',return_value=READY), \
             patch('util.gpu_wait.fcntl.flock',side_effect=BlockingIOError()), \
             patch('util.gpu_wait.time.sleep',side_effect=KeyboardInterrupt()),contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(KeyboardInterrupt):
                with reserve_gpu(GpuWaitPolicy(),self.root):self.fail()
        self.assertIn('another UAV launcher',output.getvalue())

    def test_worker_is_pinned_to_the_checked_uuid_without_changing_parent_environment(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1'}):
            env=worker_environment(READY)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'GPU-test')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'1')

    def test_invalid_wait_settings_and_non_allocation_errors_are_rejected(self):
        for kwargs in [dict(min_free_mib=0),dict(poll_seconds=0),dict(poll_seconds=float('nan')),
                       dict(timeout_seconds=-1),dict(timeout_seconds=float('inf')),dict(oom_retries=-1),dict(gpu_id='0,1')]:
            with self.assertRaises(ValueError):GpuWaitPolicy(**kwargs)
        for error in [RuntimeError('CUDA out of memory. Tried to allocate 2 MiB'),
                      RuntimeError('CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)')]:
            self.assertTrue(is_cuda_allocation_failure(error))
        for error in [ValueError('CUDA out of memory'),RuntimeError('CUBLAS_STATUS_EXECUTION_FAILED'),
                      RuntimeError('CPU allocator out of memory'),RuntimeError('invalid class label')]:
            self.assertFalse(is_cuda_allocation_failure(error))


class SuiteResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        config=self.root/'mask_ce.yaml';config.write_text('train: {}\nevaluation: {}\n')
        self.state={'cases':{'mask_ce':{'config':str(config)}},'jobs':{}}

    @contextlib.contextmanager
    def ready(self,policy):
        yield READY

    def finish(self,command,**kwargs):
        result=Path(command[command.index('--result')+1]);run=result.parent/'run';run.mkdir()
        suite.save(run/'run.json',{'status':'complete'})
        suite.save(result,{'run_dir':str(run)})

    def test_wait_occurs_before_worker_creation_and_queued_state_is_saved(self):
        @contextlib.contextmanager
        def interrupted(policy):
            state=json.loads((self.root/'suite.json').read_text())
            self.assertEqual(state['jobs']['mask_ce']['status'],'waiting')
            self.assertEqual(list(self.root.glob('mask_ce_*')),[])
            raise KeyboardInterrupt()
            yield
        with patch.object(suite,'reserve_gpu',interrupted),patch.object(suite.subprocess,'run') as run:
            with self.assertRaises(KeyboardInterrupt):suite.run_cases(self.root,self.state,'mask_ce',GpuWaitPolicy())
        run.assert_not_called()
        self.assertEqual(self.state['jobs']['mask_ce']['status'],'interrupted')

    def test_cuda_resource_failure_requeues_in_fresh_worker_and_retains_attempts(self):
        def launch(command,**kwargs):
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'GPU-test')
            result=Path(command[command.index('--result')+1])
            if run.call_count==1:
                suite.save(result,{'status':'failed','cuda_allocation_failure':True})
                raise subprocess.CalledProcessError(1,command)
            self.finish(command,**kwargs)
        with patch.object(suite,'reserve_gpu',self.ready),patch.object(suite,'artifacts',lambda run:[run/'run.json']), \
             patch.object(suite.subprocess,'run',side_effect=launch) as run,patch.object(suite.time,'sleep') as sleep, \
             contextlib.redirect_stdout(io.StringIO()):
            suite.run_cases(self.root,self.state,'mask_ce',GpuWaitPolicy())
        self.assertEqual(run.call_count,2);sleep.assert_called_once_with(30.)
        job=self.state['jobs']['mask_ce'];self.assertEqual(job['status'],'complete');self.assertNotIn('error',job)
        self.assertEqual([a['status'] for a in job['attempts']],['failed','complete'])
        self.assertNotEqual(job['attempts'][0]['attempt'],job['attempts'][1]['attempt'])

    def test_persistent_oom_has_a_retry_limit_instead_of_an_endless_training_loop(self):
        def launch(command,**kwargs):
            suite.save(Path(command[command.index('--result')+1]),{'cuda_allocation_failure':True})
            raise subprocess.CalledProcessError(1,command)
        with patch.object(suite,'reserve_gpu',self.ready),patch.object(suite.subprocess,'run',side_effect=launch) as run, \
             patch.object(suite.time,'sleep'),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                suite.run_cases(self.root,self.state,'mask_ce',GpuWaitPolicy(oom_retries=1))
        self.assertEqual(run.call_count,2);self.assertEqual(self.state['jobs']['mask_ce']['status'],'failed')
        self.assertTrue(all(a['status']=='failed' for a in self.state['jobs']['mask_ce']['attempts']))

    def test_other_errors_and_missing_markers_fail_without_retry(self):
        for marker in [None,False,'true']:
            def launch(command,**kwargs):
                if marker is not None:
                    suite.save(Path(command[command.index('--result')+1]),{'cuda_allocation_failure':marker})
                raise subprocess.CalledProcessError(1,command)
            self.state['jobs']={}
            with patch.object(suite,'reserve_gpu',self.ready),patch.object(suite.subprocess,'run',side_effect=launch) as run:
                with self.assertRaises(subprocess.CalledProcessError):suite.run_cases(self.root,self.state,'mask_ce',GpuWaitPolicy())
            self.assertEqual(run.call_count,1)

    def test_worker_writes_typed_failure_marker_and_preserves_original_exception(self):
        for error,expected in [(RuntimeError('CUDA out of memory'),True),(ValueError('bad labels'),False)]:
            result=self.root/'result.json'
            with patch('util.cli.main',side_effect=error):
                with self.assertRaises(type(error)):suite.worker(self.root/'mask_ce.yaml',result)
            self.assertEqual(json.loads(result.read_text())['cuda_allocation_failure'],expected)

    def test_worker_subcommand_reaches_the_worker_function(self):
        config=self.root/'mask_ce.yaml';result=self.root/'result.json'
        with patch.object(suite,'worker') as worker:
            suite.main(['_worker','--config',str(config),'--result',str(result)])
        worker.assert_called_once_with(config,result)

    def test_real_worker_process_preserves_a_configuration_failure_marker(self):
        result=self.root/'worker_failure.json'
        process=subprocess.run([sys.executable,str(Path(suite.__file__).resolve()),'_worker',
            '--config',str(self.root/'missing_config.yaml'),'--result',str(result)],
            capture_output=True,text=True,timeout=20)
        self.assertNotEqual(process.returncode,0)
        payload=json.loads(result.read_text())
        self.assertFalse(payload['cuda_allocation_failure'])
        self.assertIn('FileNotFoundError',payload['error'])
        self.assertIn('FileNotFoundError',process.stderr)

    def test_wait_timeout_marks_failure_without_launching_any_attempt(self):
        @contextlib.contextmanager
        def timeout(policy):
            raise TimeoutError('No free GPU memory')
            yield
        with patch.object(suite,'reserve_gpu',timeout),patch.object(suite.subprocess,'run') as run:
            with self.assertRaises(TimeoutError):suite.run_cases(self.root,self.state,'mask_ce',GpuWaitPolicy())
        run.assert_not_called();job=self.state['jobs']['mask_ce']
        self.assertEqual(job['status'],'failed');self.assertEqual(job['attempts'],[])

    def test_no_wait_mode_keeps_original_execution_and_does_not_query_nvidia(self):
        with patch.object(suite,'reserve_gpu',side_effect=AssertionError('GPU query should be opt in')), \
             patch.object(suite,'artifacts',lambda run:[run/'run.json']), \
             patch.object(suite.subprocess,'run',side_effect=self.finish) as run:
            suite.run_cases(self.root,self.state,'mask_ce')
        self.assertNotIn('env',run.call_args.kwargs)
        self.assertEqual(self.state['jobs']['mask_ce']['status'],'complete')

    def test_conflicting_model_device_is_rejected_before_waiting(self):
        config=Path(self.state['cases']['mask_ce']['config'])
        for text in ['train: {yolo_device: "1"}\n','evaluation: {device: "cpu"}\n']:
            config.write_text(text)
            with patch.object(suite,'reserve_gpu') as wait:
                with self.assertRaises(ValueError):suite.run_cases(self.root,self.state,'mask_ce',GpuWaitPolicy())
            wait.assert_not_called()


if __name__=='__main__':unittest.main()
