from fasttext import load_model

# original BIN model loading
f = load_model("isl_fasttext_rmh_fb.bin")
lines=[]

# get all words from model
words = f.get_words()

with open('isl_2_fasttext_rmh_fb.vec','w') as file_out:
    
    # the first line must contain number of total words and vector dimension
    file_out.write(str(len(words)) + " " + str(f.get_dimension()) + "\n")

    # line by line, you append vectors to VEC file
    for w in words:
        v = f.get_word_vector(w)
        vstr = ""
        for vi in v:
            vstr += " " + str(vi)
        try:
            file_out.write(w + vstr+'\n')
        except:
            pass
