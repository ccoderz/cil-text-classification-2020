import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file", help="filename to the txt file", type=str)
args = parser.parse_args()

filename = args.file

maxi_c = 0
maxi_w = 0
maxline_c = ""
maxline_w = ""
with open(filename) as f:
	data = f.readlines()
for line in data:
	if len(line) > maxi_c:
		maxi_c = len(line)
		maxline_c = line
	words = line.rstrip().split()
	if len(words) > maxi_w:
		maxi_w = len(words)
		maxline_w = line

print("max number of characters:", maxi_c, maxline_c)
print("max number of words:", maxi_w, maxline_w)

