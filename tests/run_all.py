"""Run every headless test:  python tests/run_all.py"""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
for t in ["smoke_test.py", "test_dependency_logic.py", "test_documents.py", "test_search_loop.py", "test_sources.py", "test_library_flow.py", "run_app_with_stub.py", "run_app_with_stub.py app_advanced.py"]:
    print("=" * 20, t)
    r = subprocess.run([sys.executable] + [os.path.join(here, x) if x.endswith("stub.py") or x.endswith("test.py") or x.endswith("logic.py") or x.endswith("documents.py") or x.endswith("loop.py") or x.endswith("sources.py") or x.endswith("flow.py") else x for x in t.split()], capture_output=True, text=True)
    print(r.stdout[-600:]); 
    if r.returncode:
        print(r.stderr); sys.exit(1)
print("ALL TESTS PASSED")
