import os, sys
with open(sys.argv[0]) as f, open(sys.argv[0]+'.rev','w') as o_f:
	for row in f:
		img = f[0:-3]
		if not os.path.exists(img):
			print(img)
			continue
		o_f.write(row)