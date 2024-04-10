# RealmDreamer: Text-Driven 3D Scene Generation with Inpainting and Depth Diffusion [Arxiv 2024]

[[Project Page]](https://realmdreamer.github.io/)

>  We introduce RealmDreamer, a technique for generation of general forward-facing 3D scenes from text descriptions. Our technique optimizes a 3D Gaussian Splatting representation to match complex text prompts. We initialize these splats by utilizing the state-of-the-art text-to-image generators, lifting their samples into 3D, and computing the occlusion volume. We then optimize this representation across multiple views as a 3D inpainting task with image-conditional diffusion models. To learn correct geometric structure, we incorporate a depth diffusion model by conditioning on the samples from the inpainting model, giving rich geometric structure. Finally, we finetune the model using sharpened samples from image generators. Notably, our technique does not require training on any scene-specific dataset and can synthesize a variety of high-quality 3D scenes in different styles, consisting of multiple objects. Its generality addi- tionally allows 3D synthesis from a single image.

# Code Release

We hope to put out the code around June.

# Citation

If you find our work interesting, please consider citing us!

<pre>
      @article{shriram2024realmdreamer,
        title={RealmDreamer: Text-Driven 3D Scene Generation with 
                Inpainting and Depth Diffusion},
        author={Jaidev Shriram and Alex Trevithick and Lingjie Liu and Ravi Ramamoorthi},
        journal={arXiv},
        year={2024}
    }
</pre>
