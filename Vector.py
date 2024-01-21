import math

class Vector(object):

  @classmethod
  def Lerp(cls, a, b, s):
    return (1.0-s)*a + s*b

  @classmethod
  def Cross(cls, a, b):
    return a^b

  def __init__(self, *components):
    if len(components) == 1:
      self.components = list(components[0])
    else:
      self.components = list(map(float, components))
    if len(self.components) < 3:
      self.components.extend((0,)*3)

  @property
  def x(self):
    return self.components[0]

  @x.setter
  def x(self, value):
    self.components[0] = value

  @property
  def y(self):
    return self.components[1]

  @y.setter
  def y(self, value):
    self.components[1] = value

  @property
  def z(self):
    return self.components[2]

  @z.setter
  def z(self, value):
    self.components[2] = value

  def __getitem__(self, index):
    return self.components[index]

  def __setitem__(self, index, value):
    self.components[index] = value

  def __len__(self):
    return len(self.components)

  def __iter__(self):
    return iter(self.components)

  def __add__(self, other):
    return Vector(*(a+b for a,b in zip(self, other)))

  def __mul__(self, other):
    try:
      return sum(a*b for a,b in zip(self, other))
    except:
      return Vector(*(a*other for a in self))

  def __rmul__(self, other):
    return self.__mul__(other)

  def __radd__(self, other):
    return self.__add__(other)

  def __sub__(self, other):
    if isinstance(other, Vector):
      return Vector(*(a - b for a, b in zip(self, other)))
    else:
      # Handling subtraction with a scalar or other types if necessary
      raise TypeError(f"Subtraction not supported between instances of 'Vector' and '{type(other)}'")

  def __rsub__(self, other):
    if isinstance(other, Vector):
      return Vector(*(b - a for a, b in zip(self, other)))
    else:
      # Handling right-hand side subtraction with a scalar or other types if necessary
      raise TypeError(f"Subtraction not supported between instances of '{type(other)}' and 'Vector'")

  def __neg__(self, other):
    return Vector(*(-a for a in self))

  def __str__(self):
    return '<{}>'.format(', '.join(map(str, self)))

  def __eq__(self, other):
    return tuple(self) == tuple(other)

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash(tuple(self))

  def __repr__(self):
    return str(self)

  def __xor__(a, b):
    return Vector(a.y*b.z - a.z*b.y,
                  a.z*b.x - a.x*b.z,
                  a.x*b.y - a.y*b.x)

  def __truediv__(self, other):
    if isinstance(other, (float, int)):
        return Vector(*(a / other for a in self))
    raise TypeError("Unsupported operand type(s) for /: 'Vector' and '{}'".format(type(other).__name__))

  @property
  def mag2(self):
    return sum(a * a for a in self)

  @property
  def mag(self):
    return math.sqrt(self.mag2)

  @property
  def normalized(self):
    mag2 = self * self
    if mag2 == 0:
        return Vector()
    return Vector(*self) / math.sqrt(self.mag2)

V = Vector # For convenience.