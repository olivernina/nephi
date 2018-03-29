import Levenshtein

import sys  
stdout = sys.stdout
reload(sys)  
sys.setdefaultencoding('utf-8')
sys.stdout = stdout
encoding="utf-8"

def cer(r, h):
    # Extra space would be errors for this competition
    #r = ' '.join(r.split())
    #h = ' '.join(h.split())
    
    # I think the extra conditions should be able to make sure only strings or only unicode are used here
    if (r == None and isinstance(h, str)):
        r = ''
    elif (r == None):
        r = u'' 
    if (h == None and isinstance(r, str)):
        h = ''
    elif (h == None):
        h = u''   
    
    #print("Debug: Starting a new comparison")
    #print("Debug: Printing the prediction")
    #print(r)
    #print("Debug: Is it a string? %s" % str(isinstance(r, str)))
    #print("Debug: Printing the ground truth")
    #print(h)
    #print("Debug: Is it a string? %s" % str(isinstance(h, str)))
    
    # The first error occurred when the prediction was either blank or it was a space, and the ground truth was um and was a unicode. I think that just setting everything to unicode here may be able to save it.
    
    # This hit an error when there were too many words in a line for the char encoding
    if (isinstance(r, str)):
        r = unicode(r, encoding)
    if (isinstance(h, str)):
        h = unicode(h, encoding)
    
    # Levenshtein supports unicode strings so no need to write encode('utf-8')
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
    
    # This could be a problem if there are a ton of words in a sequence, as the char code. But it should work now for training the system with or without unicode.
    # With this change the code ran through without error. However, it only went through the validation set, saved output and quit with NO ERRORS (!:)). It did not predict on the training set, it seems. Interestingly, the running average loss was about the same as the total loss. The character error rate was 0.74 on training and 0.82 on validation so a lot closer now!
    current_char = 170 # ad-hoc search of valid unicode for this, this will at least allow to go up to 400 so should have plenty of words for each line in any dataset we work with
    for w in r + h:
        if w not in word_lookup:
            word_lookup[w] = unichr(current_char)
            current_char += 1

    r = "".join([word_lookup[w] for w in r])
    h = "".join([word_lookup[w] for w in h])

    return cer(r,h)