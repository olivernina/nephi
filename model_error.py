import Levenshtein
def cer(r, h):
    r = u' '.join(r.split())
    h = u' '.join(h.split())
    steps = Levenshtein.editops(r, h)
    d = len([v for v in steps if v[0] == 'delete'])
    i = len([v for v in steps if v[0] == 'insert'])
    s = len([v for v in steps if v[0] == 'replace'])
    if len(r) == 0.0:
        return 0.0
    res = float(i+d+s) / float(len(r))
    return res

def wer(r, h):
    r = r.split()
    h = h.split()
    word_lookup = {}

    current_char = ord('a')
    for w in r + h:
        if w not in word_lookup:
            word_lookup[w] = chr(current_char)
            current_char += 1

    r = u"".join([word_lookup[w] for w in r])
    h = u"".join([word_lookup[w] for w in h])

    return cer(r,h)