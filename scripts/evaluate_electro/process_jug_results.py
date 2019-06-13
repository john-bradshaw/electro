import jug
import jug.task



jug.init('eval_electro_jug.py', 'eval_electro_jug.jugdata')
import eval_electro_jug


results = jug.task.value(eval_electro_jug.results)

# Now use these results
import numpy as np
RESULTS_FROM_JUG = 'results_from_jug.txt'

# Stitch the results back together:
top_k_accs, result_lines = zip(*results)
acc_storage = np.stack(top_k_accs)

# We now compute the path level average accuracies and print these out.
top_k_accs = np.mean((np.cumsum(acc_storage, axis=1) > 0.5).astype(np.float64), axis=0)
for k, k_acc in enumerate(top_k_accs, start=1):
    print(f"The top-{k} accuracy is {k_acc}")

# Finally we store the reaction paths in a text file.
with open(RESULTS_FROM_JUG, 'w') as fo:
    fo.writelines('\n'.join(result_lines))