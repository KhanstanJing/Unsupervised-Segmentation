import os.path

from skimage import segmentation, color, io
from skimage import graph
from matplotlib import pyplot as plt

def seg(input_folder, name, output_folder):
    input_path = os.path.join(input_folder, name)

    img = io.imread(input_path)

    labels1 = segmentation.slic(img, compactness=240, n_segments=120, start_label=1, channel_axis=None)

    g = graph.rag_mean_color(img, labels1)
    labels2 = graph.cut_threshold(labels1, g, 29)
    out2 = color.label2rgb(labels2, img, kind='avg', bg_label=0)

    output_path = os.path.join(output_folder, name)

    plt.imshow(out2)
    plt.axis("off")
    # plt.show()
    plt.savefig(output_path)