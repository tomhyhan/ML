## Calculus

**Hard problem => Sum of many small values**

d = Basically think of it as small difference- i.e. tiny change in value (x, A ... )

dA = height \* width
=> dA = x2 \* dx
=> dA / dx = f(x) (the height of the graph at that point)

## Derivative

**Tiny change in the value of a function divided by the tiny change in the input that caused it is what the `derivative` is**

dA / dx is called a derivative of A

ds / dt = rise / run = d distance / d time

Best constant approximation of rate of change around a point; which is a tangent to the slope of a function at that point.

dt / ds (t) `Velocity` = s(t+dt)−s(t) `distance` / dt `time`

Convention for `d` is that `d` is approacheing 0, and this is where magic happens! As `d` approaches 0, we only need to think about the contant value from a result of `dt / dx (t)` (tiny difference in rise / tiny different in run)

### Different view points of derivative

**Power Rule**

x ** n = (x + dx) ** n = (x +dx)(x +dx)(x +dx)...(x +dx) `n times`

-> x ** n `volume of original square` + n \* x ** (n-1) \* dx `portion of increase in the ouput` + (Multiple of dx \*\* 2), Then, all we care about is the `portion of increase in the ouput` as dx approaching 0

d(x\*\*2) = 2x \* dx

### Chain rule and Product rule

#### SUM RULE

d/dx \* (g(x) + h(x)) = dg/dx + dh/dx

#### Product Rule

df / dx = g(x) \* dh / dx + h(x) \* dg / dx

left d right right d left => think of it area of bot and side rectecgle

#### Function Composition

d / dx \* g(h(x)) = dg / dh \* (h(x)) \* dh / dx \* (x)

### Exponential Derivative

M(t) = 2 \*\* t

dM / dt => tiny change in mass over the tiny change in time
= (d**(t+dt) - 2**t) / dt\
= (2\*\*t \* 2\*\*dt - 2\*\*t) / dt\
= 2\*\*t((2\*\*dt-1) / dt)

as dt -> inf, `((2\*\*dt-1) / dt)` this here becomes constant
, which the constant value is proportion to itself.

ex. 2\*\*t(0,6931472...) => `itself * constant`

All other derivatives have constant value proportion to itself. HOWEVER, `e` has constant equals to `1` which means that the derivative of `e**t` is `e**t`.

THen, we can start asking:

2 = e \*\* ln(2) -> `e to the what equals 2?`

2\*\*t = e\*\*ln(2)t
takes derivative,
ln(2)2\*\*t = ln(2)e\*\*ln(2)t

derivative of e\*\*ct = ce\*\*ct

Super important to note:

All sorts of natural phenomena involve some rate of change that's proportional to the thing that's changing.

If some variable's rate of change is proportion to itself, the function describing that variable over time is going to look like some kind of exponential

e\*\*ct where c is the proportionality constant

# Implicit Differentiation

`Implicit` curve is the set of all points (x,y) that satisfy some property written in terms of the two variables x and y.

x\*\*2 + y\*\*2 = 5\*\*2

2x \* dx + 2y \* dy = 0 `implicit differentiation`, but why?

Think first related rates,

x(t)\*\*2 + y(t)\*\*2 = 5\*\*2, where the left equation is the function of time that happens to be constant becuase the value 5\*\*2 does not change while the time passes

2x(t)dx/dt + 2y(t)dy/dt = 0

,but our circle relationship equation does not have any `t` that ties two variables

dS = 2x \* dx + 2y \* dy

what it means to take a derivative of this function is that some tiny change in `x` and `y`to the change in `S`.

Important: whatever tiny step with dx, dy, if it's going to keep us on the curve, the values of both the left-side and right-hand side must change by the same amount

# LiMiTs

Formal definition of derivative:

df/dx = lim[h -> 0] (f(2 + h) - f(2)) / h

Limits actually let us `avoid` talking about infinitely small change, and instead focus on `concrete finite tiny change`.

## Epsilon delta

The distance away from limiting point is called `epsilon`
The distance away from inputs is called `delta`

For the limit to exist, we will always be able to find a range of inputs around our limiting input, some distance delta around 0, so that any input within a distance delta of 0 corresponds to an output within distance epsilon of 12.

## L'Hôpital's rule

It gives the precise value for the limit whenever we come across the expresion that devides by 0.

# Integral (Integration)

Integral[0-8] v(t) dt

As dt get smaller, our integral function represents more of the reality.
dt represents two things.

1. a factor in each quantity (width) we are adding up
2. spacing between each sampled time step

s(t) => distance vs time function which is also distance in velocity vs time function

ds = dT \* v(T)
ds / dT = v(T) `super general`

**Fundamental theorem of calculus**

integral[a->b] f(x) dx = F(b) - F(a)

dF/dx \* (x) = f(x) `antiderivative of f(x)`

Anytime we want to integrate some function - and remember we think of as adding up the values f(x)dx for inputs in a certain range then asking what that sum approaches as dx approachese 0 - the first step to evaluating that integral is to find an antiderivative, some other function, `capital F(x)`, whose derivative is the ting inside the integral (`area`). Then the ingetral equals this anti derivative evaluated at the top bound, minus its value at the bottom bound.
