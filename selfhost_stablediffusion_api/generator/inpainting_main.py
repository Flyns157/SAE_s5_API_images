import PIL.Image
from selfhost_stablediffusion_api.generator.inpainting_setup import generate_mask_image, object_addition_with_instruct_inpainting, object_removal_with_instruct_inpainting, pipe

class Inpainting:

    @staticmethod
    def inpainting_choice(strategy, prompt, init_image, mask_image=None):
        # Generate mask if not provided
        if mask_image is None:
            mask_image = generate_mask_image(init_image=init_image, mask_prompt=[prompt])
        
        # Use match-case for strategy selection
        match strategy:
            case 1:
                return object_addition_with_instruct_inpainting(pipe, init_image, mask_image, prompt)
            case 2:
                return object_removal_with_instruct_inpainting(pipe, init_image, mask_image, negative_prompt=prompt)
            case 3:
                return object_addition_with_instruct_inpainting(pipe, init_image, mask_image, prompt)

    @staticmethod
    def run_test():
        # Define initial test image and mask
        init_image = PIL.Image.open("./resources/dog.png")
        mask_image = PIL.Image.open("./resources/dog_mask.png")

        # Test object addition with a provided mask
        prompt = "Make the dog bigger"
        strategy = 1
        print("Object addition with provided mask:")
        print(Inpainting.inpainting_choice(strategy, prompt, init_image, mask_image))

        # Test object addition without a provided mask (auto-generated)
        print("Object addition without provided mask:")
        print(Inpainting.inpainting_choice(strategy, prompt, init_image))

        # Test object removal
        prompt = "Remove the dog"
        strategy = 2
        print("Object removal:")
        print(Inpainting.inpainting_choice(strategy, prompt, init_image, mask_image))

        # Test object addition (different prompt)
        prompt = "Put a small robot"
        strategy = 3
        print("Object addition with a new prompt:")
        print(Inpainting.inpainting_choice(strategy, prompt, init_image, mask_image))
