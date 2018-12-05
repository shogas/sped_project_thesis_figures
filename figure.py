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
    def __init__(self, data, angle=0):
        self.data = data
        self.angle = angle


    def write(self, figure_filename, _):
        if isinstance(self.data, str):
            image_filename = self.data
        else:
            image_filename = figure_filename.replace('.tex', '.png')
            save_image(image_filename, self.data)

        if self.angle in (-90, 90):
            include_graphics_arguments = r'height=0.98\textwidth, angle={}'.format(self.angle)
        elif self.angle == 0:
            include_graphics_arguments = r'width=0.98\textwidth'
        else:
            include_graphics_arguments = r'width=0.98\textwidth, angle={}'.format(self.angle)

        return r"""\node[anchor=south west, inner sep=0pt] (img) at (0,0)%
    {{\includegraphics[{}]{{{}}}}};%
""".format(
        include_graphics_arguments,
        os.path.basename(image_filename))


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
    \fill [fill=black, fill opacity=0.5] (0.2em,0.2em) rectangle ({sb_width}/{im_width}+0.05,1.8em);
    \draw [white, line width=0.2em] (0.4em,0.6em) -- node[above,inner sep=0.1em, font=\footnotesize] {{{text}}} ({sb_width}/{im_width}+0.04, 0.6em);
\end{{scope}}
""".format(text=self.text, sb_width=self.scalebar_width_physical, im_width=self.image_width_physical)


class TikzAxis(TikzElement):
    def __init__(self, *args, **kwargs):
        self.axis_properties = kwargs
        self.elements = args


    def write(self, figure_filename, last_shape):
        result = r"""\begin{{axis}}[
    {}]
""".format(',\n    '.join(('{}={}'.format(key.replace('_', ' '), value) for key, value in self.axis_properties.items())))
        result += '    ' + (
                '\n'.join(element.write(figure_filename, last_shape) for element in self.elements)).replace('\n', '\n    ')
        result += '\n\\end{axis}'
        return result


class TikzTablePlot(TikzElement):
    def __init__(self, xs, ys, errors=[], **kwargs):
        if len(xs) != len(ys) or (len(errors) > 0 and len(xs) != len(errors)):
            raise ValueError('X-axis, y-axis and errors (if specified) must have the same length. Got {}, {}{}'.format(
                len(xs), len(ys), ', {}'.format(len(errors)) if len(errors) > 0 else ''))
        self.xs = xs
        self.ys = ys
        self.errors = errors
        self.plot_properties = kwargs


    def write(self, figure_filename, last_shape):
        result = r"""\addplot+[
    {}""".format((',\n    '.join('{}={}'.format(key.replace('_', ' '), value) for key, value in self.plot_properties.items())))
        if len(self.errors) > 0:
            result += r""",
    error bars/.cd,
        y fixed,
        y dir=both,
        y explicit] table [x=x, y=y,y error=error, col sep=comma] {{
        x,  y,       error
{}
}};""".format('\n'.join('        {}, {}, {}'.format(x, y, error) for x, y, error in zip(self.xs, self.ys, self.errors)))
        else:
            result += r"""] table [x=x, y=y, col sep=comma] {{
        x,  y,
{}
}};""".format('\n'.join('        {}, {}'.format(x, y) for x, y in zip(self.xs, self.ys)))

        return result


class TikzLegend(TikzElement):
    def __init__(self, *args):
        self.entries = args


    def write(self, figure_filename, last_shape):
        return '\n'.join(r'\addlegendentry{{{}}}'.format(entry) for entry in self.entries)

