import os

from PIL import Image


def save_image(filename, data):
    Image\
        .fromarray(data)\
        .save(filename)


def save_figure(filename, *args):
    tikz_output = ''
    last_shape = (1, 1)
    for element in args:
        if not isinstance(element, TikzElement):
            raise TypeError('Expected a Tikz element, got {}'.format(type(element)))
        if isinstance(element, TikzImage):
            last_shape = element.shape()
        tikz_output += element.write(filename, last_shape)

    with open(filename, 'w') as f:
        f.write(tikz_output)


class TikzElement:
    pass


class TikzImage(TikzElement):
    def __init__(self, data):
        self.data = data


    def write(self, figure_filename, _):
        if isinstance(self.data, str):
            image_filename = self.data
        else:
            image_filename = figure_filename.replace('.tex', '.png')
            save_image(image_filename, self.data)

        return r"""\node[anchor=south west, inner sep=0pt] (img) at (0,0)%
    {{\includegraphics[width=0.98\textwidth]{{{}}}}};%
""".format(os.path.basename(image_filename))


    def shape(self):
        return (1, 1) if isinstance(self.data, str) else self.data.shape


class TikzCircle(TikzElement):
    def __init__(self, cx, cy, r, draw_properties):
        self.cx = cx
        self.cy = cy
        self.r = r
        self.props = draw_properties


    def write(self, figure_filename, last_shape):
        rel_x = self.cx / (0.5*last_shape[0]) - 1
        rel_y = self.cy / (0.5*last_shape[1]) - 1
        rel_r = self.r / (0.5*last_shape[0])
        return r"""\draw[{}] let
    \p1 = ($(img)!{}!(img.east)$),
    \p2 = ($(img)!{}!(img.north)$),
    \p3 = ($(img.east) - (img.west)$)
    in (\x1, \y2) circle ({{veclen(\x3,\y3)*{}}});
""".format(self.props, rel_x, rel_y, rel_r)


class TikzScalebar(TikzElement):
    def __init__(self, scalebar_width_physical, image_width_physical, text):
        self.scalebar_width_physical = scalebar_width_physical
        self.image_width_physical = image_width_physical
        self.text = text


    def write(self, figure_filename, last_shape):
        return r"""\begin{{scope}}[x={{(img.south east)}},y={{(img.north west)}}]
    \fill [fill=black, fill opacity=0.5] (0.03,0.03) rectangle ({sb_width}/{im_width}+0.05,2.0em+0.05);
    \draw [white, line width=0.2em] (0.04,0.04) -- node[above,inner sep=0.1em, font=\footnotesize] {{{text}}} ({sb_width}/{im_width}+0.04,0.04);
\end{{scope}}
""".format(text=self.text, sb_width=self.scalebar_width_physical, im_width=self.image_width_physical)


