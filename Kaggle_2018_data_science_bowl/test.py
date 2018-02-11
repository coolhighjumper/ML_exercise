from PIL import Image, ImageOps

desired_size = 800
im_pth = "./bde3727f3a9e8b2b58f383ebc762b2157eb50cdbff23e69b025418b43967556b.png"

im = Image.open(im_pth)
old_size = im.size  # old_size[0] is in (width, height) format
ratio = float(desired_size)/max(old_size)
new_size = tuple([int(x*ratio) for x in old_size])
# use thumbnail() or resize() method to resize the input image
# thumbnail is a in-place operation
# im.thumbnail(new_size, Image.ANTIALIAS)
im = im.resize(new_size, Image.ANTIALIAS)
# create a new image and paste the resized on it
new_im = Image.new("RGB", (desired_size, desired_size))
new_im.paste(im, ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2))

new_im.show()
