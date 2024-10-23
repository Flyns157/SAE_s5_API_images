import PIL.Image

from inpaiting_setup import *


class Inpaiting :

    @staticmethod
    def inpaiting_choice(strategy, prompt, init_image, mask_image=None):
        if mask_image is None:
            mask_image = generate_mask_image(init_image=init_image, mask_prompt=[prompt])
        match strategy:
            case 1:
                return object_addition_with_instruct_inpainting(pipe=pipe,prompt=prompt, init_image=init_image,
                                                                mask_image=mask_image)
            case 2:
                return object_removal_with_instruct_inpainting(pipe=pipe, init_image=init_image, mask_image=mask_image,
                                                               negative_prompt=prompt)
            case 3:
                return object_addition_with_instruct_inpainting(pipe=pipe, init_image=init_image, mask_image=mask_image,
                                                                prompt=prompt)

    @staticmethod
    def test():
        init_image = PIL.Image.open("./resources/dog.png")
        mask_image = PIL.Image.open("./resources/dog_mask.png")
        prompt = "Make the dog bigger"
        strategy = 1

        print("Local editing with provide a mask")
        print(Inpaiting.inpaiting_choice(strategy,prompt,init_image,mask_image))

        print("Local editing without provide a mask")
        print(Inpaiting.inpaiting_choice(strategy,prompt,init_image))

        prompt = "Remove the dog"
        strategy=2

        print("Object removal")
        print(Inpaiting.inpaiting_choice(strategy, prompt, init_image, mask_image))

        prompt = "Put a small robot"
        strategy = 3

        print("Object additional")
        print(Inpaiting.inpaiting_choice(strategy, prompt, init_image, mask_image))