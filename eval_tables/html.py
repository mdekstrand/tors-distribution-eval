from io import StringIO
from base64 import b64encode

import numpy as np

from .sparkline import SparkKDEImageWriter


def write_html(cols, rows, sys_label, scale):
    out = StringIO()
    print('<table>', file=out)
    print('<thead><tr>', file=out)
    print('<th>{}'.format(sys_label), file=out)
    for s in cols:
        print('<th>{}'.format(s), file=out)
    print('<th>Dist. (KDE)', file=out)

    print('</tr></thead>', file=out)
    print('<tbody>', file=out)

    for row in rows:
        print('<tr><td>{}'.format(row.name), file=out)
        
        for sr in row.stats:
            print('<td>%.3f' % (sr.value,), file=out)
        
        w = SparkKDEImageWriter(row.kde, scale=scale)
        w.begin(3)
        w.point(np.mean(row.data))
        w.render_kde()
        w.end()

        ibs = w.get_content()
        print('<td><img src="data:image/png;base64,{}">'.format(b64encode(ibs).decode('ascii')), file=out)

        print('</tr>', file=out)

    print('</tbody>', file=out)
    print('</table>', file=out)

    return out.getvalue()
