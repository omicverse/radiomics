"""Execute the four tutorials in isolated output, using the current Python kernel."""
from pathlib import Path
import argparse
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager, KernelSpec

ROOT = Path(__file__).resolve().parents[1]


class CurrentPythonKernel(KernelSpecManager):
    def get_kernel_spec(self, kernel_name):
        return KernelSpec(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                          display_name='Validation Python', language='python')


parser = argparse.ArgumentParser()
parser.add_argument('--output-root', default=str(ROOT/'artifacts'))
args = parser.parse_args()
execution_root = Path(args.output_root).resolve()
destination = execution_root/'executed_notebooks'
destination.mkdir(parents=True, exist_ok=True)
for file in sorted((ROOT/'examples').glob('0*.ipynb')):
    notebook = nbformat.read(file,as_version=4)
    # Keep tutorial defaults intact; isolate execution artifacts under ignored artifacts/.
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.source = cell.source.replace('Path("output/notebook_demo")', f'Path({str(execution_root / "notebook_demo")!r})')
    client = NotebookClient(notebook, timeout=180, kernel_name='python3',
        resources={'metadata':{'path':str(ROOT/'examples')}})
    from jupyter_client import KernelManager
    client.km = KernelManager(kernel_name='python3', kernel_spec_manager=CurrentPythonKernel())
    print('Executing',file.name,flush=True)
    client.execute()
    nbformat.write(notebook,destination/file.name)
    print('Passed',file.name,flush=True)
