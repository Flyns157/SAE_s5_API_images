from inpaiting_setup import *


def inpaiting_choice(strategy, prompt, init_image, mask_image=None):
    if mask_image is None:
        mask_image = generate_mask_image(init_image=init_image, mask_prompt=[prompt])
    match strategy:
        case 1:
            return local_modification_inpaiting(prompt=prompt, init_image=init_image, mask_image=mask_image)
        case 2:
            return object_removal_with_instruct_inpainting(pipe=pipe, init_image=init_image, mask_image=mask_image,
                                                           negative_prompt=prompt)
        case 3:
            return object_addition_with_instruct_inpainting(pipe=pipe, init_image=init_image, mask_image=mask_image,
                                                            prompt=prompt)
