import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, nargs='?', const='latest', default=None)

# Test 1: no arguments
print("Test 1:", parser.parse_args([]))
# Test 2: --resume without value
print("Test 2:", parser.parse_args(['--resume']))
# Test 3: --resume with value latest
print("Test 3:", parser.parse_args(['--resume', 'latest']))
# Test 4: --resume with custom path
print("Test 4:", parser.parse_args(['--resume', '/some/path.pth']))
