try:
    from IPython.display import Image, display
    from ipywidgets import interactive, IntSlider

    IPYTHON_AVAILABLE = True
except ImportError:
    Image = None
    display = None
    interactive = None
    IntSlider = None
    IPYTHON_AVAILABLE = False

def show_image_in_ipython(data, *args, **kwargs):
    if not IPYTHON_AVAILABLE:
        raise RuntimeError("IPython display dependencies are not installed")
    display(Image(data, *args, **kwargs))

def ipython_images_slider(image_pathes_list, slider_label="", first_arg=0):
    if not IPYTHON_AVAILABLE:
        raise RuntimeError("IPython display dependencies are not installed")
    def display_f(**kwargs):
        display(Image(image_pathes_list[kwargs[slider_label]]))
    return interactive(display_f, **{slider_label: IntSlider(min=0, max=len(image_pathes_list)-1,step=1)})

def show_ipython_images_slider(image_pathes_list, slider_label="", first_arg=0):
    if not IPYTHON_AVAILABLE:
        raise RuntimeError("IPython display dependencies are not installed")
    def display_f(**kwargs):
        display(Image(image_pathes_list[kwargs[slider_label]]))
    display(interactive(display_f, **{slider_label: IntSlider(min=0, max=len(image_pathes_list)-1,step=1)}))
