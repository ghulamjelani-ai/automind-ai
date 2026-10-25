# Brand photos for the Stable Diffusion 1.5 LoRA

Put 15 to 60 of your own photos here (products, shop, food, team), in `.jpg`, `.png` or `.webp` format.

- Use clear, well-lit photos. Square or nearly square photos work best (they are cropped to 512 x 512).
- Optional: add a caption for `cake1.jpg` in `cake1.txt`, for example `chocolate fudge cake on a cream plate, bakery counter`.
  Without a caption, `a professional marketing photo` is used.
- The trigger word (default `brandstyle`) is added to every caption. The model server adds it to every image prompt, so new pictures pick up your look.
- If your photos contain text or logos, train with `--no-flip` so they are not mirrored.

With fewer than 5 photos the notebook skips image training and uses the base SD 1.5 model.
