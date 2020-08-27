from pdf2image import convert_from_path

images = convert_from_path('example.pdf', poppler_path=r"poppler\bin")
for i, image in enumerate(images):
    fname = "image" + str(i) + ".jpg"
    image.save(fname, "JPEG")
