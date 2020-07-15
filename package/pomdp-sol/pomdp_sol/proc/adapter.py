try: # NOTE: Toggle for test use (__test__.py)
    from __head__ import *
except ModuleNotFoundError:
    pass

try: # NOTE: Toggle for package use (interface.py)
    from .__head__ import *
except ImportError:
    pass

def solver(
    opt : Dict[str, str],
    out : str = None    ,
):
    bin = DIR + SEP + SOL

    cmd(bin, opt, out)

def sarsop(
    opt : Dict[str, str],
    out : str = None    ,
):
    bin = DIR + SEP + SAR

    cmd(bin, opt, out)

    with open(opt['o'], 'r') as inp:
        cnt = inp.readlines()[3:-1]

        rep = dict((re.escape(k), v) for k, v in {
            '<Vector action="' : '',
            '" obsValue="0">'  : '\n',
            ' </Vector>'       : '\n',
        }.items())

        pat = re.compile("|".join(rep.keys()))

        txt = [pat.sub(lambda m: rep[re.escape(m.group(0))], lin)
            for lin in cnt]

    with open(opt['o'], 'w') as out:
        for lin in txt:
            out.write(lin)

        out.close()

def cmd(
    bin : str           ,
    opt : Dict[str, str],
    out : str           ,
):
    conf = pck([e for tup in [(app(k), v) for (k, v) in opt.items()] for e in tup])

    if out:
        subprocess.call([bin] + conf, stdout = open(out, 'w'))
    else:
        subprocess.call([bin] + conf)

def app(
    key : str,
) -> str:
    return '-' + key if key else None

def pck(
    lst : List[str],
) -> List[str]:
    return [e for e in lst if e]
