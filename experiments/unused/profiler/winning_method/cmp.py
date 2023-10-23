from os.path import join
import os
import filecmp

rootdir = join("results", "secondary", "profiler", "winning_method")

print(rootdir)
for subdir, files in os.walk(rootdir):
    for file in files:
        f1 = os.path.join(subdir, file)
        f2 = os.path.join(subdir, file + 1)

        if filecmp.cmp(f1, f2):
            print("diff")
