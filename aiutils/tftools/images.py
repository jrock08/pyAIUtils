import tensorflow as tf

def resize_images_like(images, reference_tensor, method=0, align_corners=False):
    """ Image resize helper.

    Resizes images to the same size as images_like.

    Args:
      images (4d tensor): Input to resize layer
        Should have shape `[batch, width, height, channels]`

      reference_tensor (4d tensor): defines shape to resize to
        Should have shape `[?, width_new, height_new, ?]`
      method: ResizeMethod.  Defaults to `ResizeMethod.BILINEAR`.
      align_corners: bool. If true, exactly align all 4 cornets of the input and
         output. Defaults to `false`.

    Returns:
      output (tensor): The resized image.
        Will have shape `[batch, width_new, height_new, channels]`
    """
    _,w,h,_ = reference_tensor.get_shape().as_list()
    return tf.image.resize_images(images, w, h, method=method, align_corners=align_corners)


def all_image_crops(tensor, crop_size=[1,32,32,3]):
    """

    No zero padding
    """

    crops = []
    print tensor.get_shape()
    b,w,h,c = tensor.get_shape().as_list()
    assert(b==1)

    print w-crop_size[1]+1
    print h-crop_size[2]+1
    print c-crop_size[3]+1

    for i in range(0,w-crop_size[1]+1):
        for j in range(0,h-crop_size[2]+1):
            for k in range(0,c-crop_size[3]+1):
                crops.append(tf.slice(tensor, [0,i,j,k], crop_size))

    return tf.concat(0, crops)

def image_crops(tensor, crop_size=[1,32,32,3], num=10):

    crops = []
    for i in range(num):
        crops.append(tf.random_crop(tensor, crop_size))

    return tf.concat(0, crops)
