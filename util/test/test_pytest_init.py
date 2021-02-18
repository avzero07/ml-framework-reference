'''
Sanity Check to Verify Pytest Installation

Other Pytest or Env Specific Testcases May be Added
'''
import subprocess

def test_check_pytest_version():
    op = subprocess.run(['pytest','--version'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    assert not op.returncode, "'pytest --version' returned errorcode {}".format(op.returncode)
    assert op.stderr.decode().rstrip() == "pytest 6.2.2"
