import gensim
# model = gensim.models.FastText.load("./rmh.w2v.model")

model = gensim.models.FastText.load("./fasttext_is_rmh/rmh.fasttext.model")

# model.save_facebook_model("isl_fb_fasttext_rmh.bin")
gensim.models.fasttext.save_facebook_model(model, "isl_fasttext_rmh_fb.bin")
