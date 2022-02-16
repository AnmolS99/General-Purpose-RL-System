import cProfile, pstats, io
from pstats import SortKey
from main import run

pr = cProfile.Profile()
pr.enable()
run()
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())