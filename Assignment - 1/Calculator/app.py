from flask import Flask, render_template, request
import ast
import math

app = Flask(__name__)

ALLOWED_FUNCTIONS = {
    "abs": abs,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": pow,
}

ALLOWED_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


class SafeExpressionVisitor(ast.NodeVisitor):
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.FloorDiv,
        ast.UAdd,
        ast.USub,
        ast.Load,
        ast.Call,
        ast.Name,
        ast.Constant,
    )

    def __init__(self, allowed_names):
        super().__init__()
        self.allowed_names = allowed_names

    def visit(self, node):
        if not isinstance(node, self.allowed_nodes):
            raise ValueError("Expression contains unsupported operations.")
        return super().visit(node)

    def visit_Name(self, node):
        if node.id not in self.allowed_names:
            raise ValueError(f"Unsupported name: {node.id}")

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed.")
        if node.func.id not in ALLOWED_FUNCTIONS:
            raise ValueError(f"Unsupported function: {node.func.id}")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported.")
        for arg in node.args:
            self.visit(arg)


def safe_eval_expr(expr, **variables):
    expr = (expr or "").strip()
    if not expr:
        raise ValueError("Function expression cannot be empty.")

    allowed_names = dict(ALLOWED_FUNCTIONS)
    allowed_names.update(ALLOWED_CONSTANTS)
    allowed_names.update(variables)

    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid expression syntax.") from exc

    SafeExpressionVisitor(allowed_names).visit(parsed)
    value = eval(compile(parsed, "<expr>", "eval"), {"__builtins__": {}}, allowed_names)

    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expression produced a non-finite value.")
    return float(value)


def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    n = len(A)
    x = [0.0] * n
    steps = []

    for i in range(n):
        if A[i][i] == 0:
            raise ValueError(f"Diagonal element A[{i+1},{i+1}] cannot be zero.")

    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sigma = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sigma) / A[i][i]

        error = max(abs(x_new[i] - x[i]) for i in range(n))
        if not all(math.isfinite(v) for v in x_new):
            raise ValueError("Iteration diverged to a non-finite value. Check your system.")

        steps.append((k + 1, x_new.copy(), error))
        x = x_new

        if error < tol:
            return x, steps, "Converged"

    return x, steps, "Max iterations reached"


def trapezoidal_from_points(points):
    if len(points) < 2:
        raise ValueError("At least two points are required.")

    x_values = [p[0] for p in points]
    f_values = [p[1] for p in points]

    h = x_values[1] - x_values[0]
    if h == 0:
        raise ValueError("x values must be strictly increasing and equally spaced.")

    for i in range(2, len(x_values)):
        if not math.isclose(x_values[i] - x_values[i - 1], h, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Table mode requires equally spaced x values.")

    n = len(points) - 1
    total = f_values[0] + f_values[-1] + 2 * sum(f_values[1:-1])
    return (h / 2) * total, h, n


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/gauss', methods=['GET', 'POST'])
def gauss():
    result = steps = status = error = None
    if request.method == 'POST':
        try:
            A = [
                [float(request.form['a11']), float(request.form['a12']), float(request.form['a13'])],
                [float(request.form['a21']), float(request.form['a22']), float(request.form['a23'])],
                [float(request.form['a31']), float(request.form['a32']), float(request.form['a33'])],
            ]
            b = [float(request.form['b1']), float(request.form['b2']), float(request.form['b3'])]

            result, steps, status = gauss_seidel(A, b)
        except ValueError as exc:
            error = str(exc)
        except Exception:
            error = "Please enter valid numeric values for all coefficients."

    return render_template("gauss.html", result=result, steps=steps, status=status, error=error)


@app.route('/trapezoidal', methods=['GET', 'POST'])
def trapezoidal():
    result = table = h = mode = error = None
    if request.method == 'POST':
        try:
            f_expr = (request.form.get('function') or "").strip()
            x_fields = [request.form.get(f'x{i}', '').strip() for i in range(5)]
            f_fields = [request.form.get(f'f{i}', '').strip() for i in range(5)]

            if f_expr:
                mode = "function"
                a = float(request.form['a'])
                b = float(request.form['b'])
                n = int(request.form['n'])

                if n <= 0:
                    raise ValueError("Number of intervals n must be greater than zero.")

                h = (b - a) / n
                total = 0.0
                table = []

                for i in range(n + 1):
                    x = a + i * h
                    fx = safe_eval_expr(f_expr, x=x)
                    table.append((x, fx))
                    total += fx if i in (0, n) else 2 * fx

                result = (h / 2) * total
            else:
                mode = "table"
                points = []
                for i in range(5):
                    if x_fields[i] and f_fields[i]:
                        points.append((float(x_fields[i]), float(f_fields[i])))
                    elif x_fields[i] or f_fields[i]:
                        raise ValueError("In table mode, each x value must have a matching f(x) value.")

                if not points:
                    raise ValueError("Enter either a function expression or at least two table points.")

                result, h, _ = trapezoidal_from_points(points)
                table = points

        except ValueError as exc:
            error = str(exc)
        except Exception:
            error = "Invalid input. Please check your values and try again."

    return render_template("trapezoidal.html", result=result, table=table, h=h, mode=mode, error=error)


@app.route('/milne', methods=['GET', 'POST'])
def milne():
    result = steps = error = None
    if request.method == 'POST':
        try:
            f_expr = request.form['function']
            x0 = float(request.form['x0'])
            y0 = float(request.form['y0'])
            h = float(request.form['h'])
            steps_count = int(request.form['steps'])

            if h == 0:
                raise ValueError("Step size h cannot be zero.")
            if steps_count < 4:
                raise ValueError("Steps must be at least 4 for Milne's method.")

            def f(x, y):
                return safe_eval_expr(f_expr, x=x, y=y)

            values = [(x0, y0)]
            x, y = x0, y0

            for _ in range(3):
                y = y + h * f(x, y)
                x = x + h
                if not math.isfinite(y):
                    raise ValueError("Non-finite y encountered while generating starter points.")
                values.append((x, y))

            while len(values) < steps_count:
                i = len(values) - 1
                x_next = values[i][0] + h
                yp = values[i - 3][1] + (4 * h / 3) * (
                    2 * f(*values[i - 2]) - f(*values[i - 1]) + 2 * f(*values[i])
                )
                yc = values[i - 1][1] + (h / 3) * (
                    f(*values[i - 1]) + 4 * f(*values[i]) + f(x_next, yp)
                )
                if not math.isfinite(yc):
                    raise ValueError("Method diverged to a non-finite value.")
                values.append((x_next, yc))

            result = values[-1]
            steps = values
        except ValueError as exc:
            error = str(exc)
        except Exception:
            error = "Invalid input. Please verify expression and numeric values."

    return render_template("milne.html", result=result, steps=steps, error=error)


if __name__ == "__main__":
    app.run(debug=True)
