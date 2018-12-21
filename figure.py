import os

import numpy as np
from PIL import Image


def save_image(filename, data):
    Image\
        .fromarray(data)\
        .save(filename)


def save_figure(filename, *elements):
    tikz_output = ''
    for element in elements:
        if not isinstance(element, TikzElement):
            raise TypeError('Expected a Tikz element, got {}'.format(type(element)))
        tikz_output += element.write(filename, elements)

    with open(filename, 'w') as f:
        f.write(tikz_output)

material_color_palette = [
    ( 'MaterialRed',        np.array((244,  67,  54)) / 255.0),
    ( 'MaterialGreen',      np.array(( 76, 175,  80)) / 255.0),
    ( 'MaterialBlue',       np.array(( 33, 150, 243)) / 255.0),
    ( 'MaterialYellow',     np.array((255, 235,  59)) / 255.0),
    ( 'MaterialPurple',     np.array((156,  39, 176)) / 255.0),
    ( 'MaterialCyan',       np.array((  0, 188, 212)) / 255.0),
    ( 'MaterialDeepPurple', np.array((103,  58, 183)) / 255.0),
    ( 'MaterialAmber',      np.array((255, 193,   7)) / 255.0),
    ( 'MaterialLightGreen', np.array((139, 195,  74)) / 255.0),
    ( 'MaterialPink',       np.array((233,  30,  99)) / 255.0),
    ( 'MaterialDeepOrange', np.array((255,  87,  34)) / 255.0),
    ( 'MaterialBrown',      np.array((121,  85,  72)) / 255.0),
    ( 'MaterialGray',       np.array((158, 158, 158)) / 255.0),
    ( 'MaterialIndigo',     np.array(( 63,  81, 181)) / 255.0),
    ( 'MaterialLightBlue',  np.array((  3, 169, 244)) / 255.0),
    ( 'MaterialLime',       np.array((205, 220,  57)) / 255.0),
    ( 'MaterialTeal',       np.array((  0, 150, 136)) / 255.0),
    ( 'MaterialOrange',     np.array((255, 152,   0)) / 255.0),
    ( 'MaterialBlueGray',   np.array(( 96, 125, 139)) / 255.0),
]

class TikzElement:
    pass


class TikzImage(TikzElement):
    def __init__(self, data, angle=0):
        self.data = data
        self.angle = angle


    def write(self, figure_filename, figure_elements):
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
        if any(el for el in figure_elements if isinstance(el, TikzColorbar) and not el.horizontal):
            include_graphics_arguments = include_graphics_arguments.replace('textwidth', 'textwidth-2cm')

        return r"""\node[anchor=south west, inner sep=0pt] (img) at (0,0)%
    {{\includegraphics[{}]{{{}}}}};%
""".format(
        include_graphics_arguments,
        os.path.basename(image_filename))


    def shape(self):
        return (1, 1) if isinstance(self.data, str) else self.data.shape


class TikzCircle(TikzElement):
    def __init__(self, cx, cy, r, draw_properties, transform=np.identity(2), text=None, text_properties=''):
        self.cx = cx
        self.cy = cy
        self.r = r
        self.transform = transform
        self.props = draw_properties
        self.text = text
        self.text_properties = text_properties


    def write(self, _, figure_elements):
        last_shape = next(el for el in figure_elements if isinstance(el, TikzImage)).shape()
        pos = self.transform @ np.array([
            self.cx / (0.5*last_shape[1]) - 1,
            self.cy / (0.5*last_shape[0]) - 1])
        rel_r = self.r / (0.5*last_shape[0])
        if self.text is not None:
            node = '  node [{}] {{{}}}'.format(self.text_properties, self.text)
        else:
            node = ''
        return r"""\draw[{}] let
    \p1 = ($(img)!{}!(img.east)$),
    \p2 = ($(img)!{}!(img.north)$),
    \p3 = ($(img.east) - (img.west)$)
    in (\x1, \y2) circle ({{veclen(\x3,\y3)*{}}}){};
""".format(self.props, *pos, rel_r, node)


class TikzArrow(TikzElement):
    def __init__(self, from_x, from_y, to_x, to_y, draw_properties, text=None, text_properties=''):
        self.from_x = from_x
        self.from_y = from_y
        self.to_x = to_x
        self.to_y = to_y
        self.props = draw_properties
        self.text = text
        self.text_properties = text_properties


    def write(self, _, figure_elements):
        last_shape = next(el for el in figure_elements if isinstance(el, TikzImage)).shape()
        from_x = self.from_x / (0.5*last_shape[1]) - 1
        from_y = self.from_y / (0.5*last_shape[0]) - 1
        to_x = self.to_x / (0.5*last_shape[1]) - 1
        to_y = self.to_y / (0.5*last_shape[0]) - 1
        if self.text is not None:
            node = '  node [{}] {{{}}}'.format(self.text_properties, self.text)
        else:
            node = ''

        return r"""\draw[->, {}] let
    \p1 = ($(img)!{}!(img.east)$),
    \p2 = ($(img)!{}!(img.north)$),
    \p3 = ($(img)!{}!(img.east)$),
    \p4 = ($(img)!{}!(img.north)$)
    in (\x1, \y2) -- (\x3, \y4){};
""".format(self.props, from_x, from_y, to_x, to_y, node)



class TikzRectangle(TikzElement):
    def __init__(self, x1, y1, x2, y2, draw_properties, transform=np.identity(2), text=None, text_properties=''):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.transform = transform
        self.props = draw_properties
        self.text = text
        self.text_properties = text_properties


    def write(self, _, figure_elements):
        last_shape = next(el for el in figure_elements if isinstance(el, TikzImage)).shape()
        rel_a = self.transform @ np.array([
            self.x1 / (0.5*last_shape[1]) - 1,
            self.y1 / (0.5*last_shape[0]) - 1])
        rel_b = self.transform @ np.array([
            self.x2 / (0.5*last_shape[1]) - 1,
            self.y2 / (0.5*last_shape[0]) - 1])
        if self.text is not None:
            node = '  node [{}] {{{}}}'.format(self.text_properties, self.text)
        else:
            node = ''
        return r"""\draw[{}] let
    \p1 = ($(img)!{}!(img.east)$),
    \p2 = ($(img)!{}!(img.north)$),
    \p3 = ($(img)!{}!(img.east)$),
    \p4 = ($(img)!{}!(img.north)$)
    in (\x1, \y2) rectangle (\x3, \y4){};
""".format(self.props, *rel_a, *rel_b, node)


class TikzScalebar(TikzElement):
    def __init__(self, scalebar_width_physical, image_width_physical, text):
        print('TikzScalebar', scalebar_width_physical, image_width_physical)
        self.scalebar_width_physical = scalebar_width_physical
        self.image_width_physical = image_width_physical
        self.text = text


    def write(self, figure_filename, figure_elements):
        return r"""\begin{{scope}}[x={{(img.south east)}},y={{(img.north west)}}]
    \fill [fill=black, fill opacity=0.5] (0.2em, 0.2em) rectangle ++ ($ ({sb_width}/{im_width}, 0) + (0.2em, 1.6em) $);
    \draw [white, line width=0.2em] (0.4em,0.6em) -- node[above,inner sep=0.1em, font=\footnotesize] {{{text}}} ++ ({sb_width}/{im_width}, 0);
\end{{scope}}
""".format(text=self.text, sb_width=self.scalebar_width_physical, im_width=self.image_width_physical)


class TikzColorbar(TikzElement):
    def __init__(self, min_value, max_value, step_value, colormap, length, horizontal=False, **kwargs):
        self.min_value = min_value
        self.max_value = max_value
        self.step_value = step_value
        self.colormap = colormap
        self.length = length
        self.horizontal = horizontal
        if len(kwargs) > 0:
            self.styles = ',\n    ' + ',\n    '.join('{}={}'.format(key.replace('_', ' '), value)
                    for key, value in kwargs.items())
        else:
            self.styles = ''


    def write(self, figure_filename, figure_elements):
        if self.horizontal:
            colorbar_direction = ' horizontal'
            size_settings = 'width={}'.format(self.length)
        else:
            colorbar_direction = ''
            size_settings = 'height={}'.format(self.length)

        return r"""\begin{{axis}}[
    hide axis,
    scale only axis,
    height=0pt,
    width=0pt,
    colorbar,
    colormap/{colormap},
    colorbar{dir},
    point meta min={min_value},
    point meta max={max_value},
    colorbar style={{
        {size}
    }}{styles}]
    \addplot [draw=none] coordinates {{(0,0) (1,1)}};
\end{{axis}}
        """.format(
                colormap=self.colormap,
                min_value=self.min_value,
                max_value=self.max_value,
                dir=colorbar_direction,
                size=size_settings,
                styles=self.styles)


class TikzAxis(TikzElement):
    def __init__(self, *args, axis_type='', **kwargs):
        self.axis_properties = kwargs
        self.axis_type = axis_type
        self.elements = args


    def write(self, figure_filename, figure_elements):
        result = r"""\begin{{{}axis}}[
    {}]
""".format(self.axis_type, ',\n    '.join(('{}={}'.format(key.replace('_', ' '), value)
                                          for key, value in self.axis_properties.items())))
        result += '    ' + (
                '\n'.join(element.write(figure_filename, figure_elements) for element in self.elements)).replace('\n', '\n    ')
        result += '\n\\end{{{}axis}}'.format(self.axis_type)
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


    def write(self, figure_filename, figure_elements):
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
        x,  y
{}
}};""".format('\n'.join('        {}, {}'.format(x, y) for x, y in zip(self.xs, self.ys)))

        return result


class TikzTable3D(TikzElement):
    def __init__(self, coords):
        self.coords = coords


    def write(self, figure_filename, figure_elements):
        return r"""\addplot3+[only marks, scatter] table [x=x, y=y, z=z, col sep=comma] {{
    x, y, z
{}
}};""".format('\n'.join('    {}, {}, {}'.format(*coord) for coord in self.coords))


class TikzPlot3D(TikzElement):
    def __init__(self, expression, **kwargs):
        self.expression = expression
        self.line_style = kwargs


    def write(self, figure_filename, figure_elements):
        return r"""\addplot3[
    mesh,
    variable = \u,
    {}]
    ({});""".format(',\n    '.join('{}={}'.format(key.replace('_', ' '), value) for key, value in self.line_style.items()), self.expression)

class TikzLegend(TikzElement):
    def __init__(self, *args):
        self.entries = args


    def write(self, figure_filename, figure_elements):
        return '\n'.join(r'\addlegendentry{{{}}}'.format(entry) for entry in self.entries)

