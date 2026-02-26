from flask import Flask, render_template, request
import math

app = Flask(__name__)

# =======================
# HOME
# =======================
@app.route('/')
def home():
    return render_template("index.html")

# =======================
# GAUSS SEIDEL METHOD
# =======================
def gauss_seidel_method(A, b, tol=0.0001, max_iter=50):
    n = len(A)
    x = [0.0] * n
    steps = []

    for k in range(max_iter):
        x_new = x.copy()

        for i in range(n):
            s = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        error = max(abs(x_new[i] - x[i]) for i in range(n))
        steps.append((k+1, x_new.copy(), error))
        x = x_new

        if error < tol:
            return x, steps, "Converged"

    return x, steps, "Max iterations reached"


@app.route('/gauss', methods=['GET', 'POST'])
def gauss():
    result = None
    steps = None
    status = None
    error_message = None

    if request.method == 'POST':
        try:
            A = [
                [float(request.form['a11']), float(request.form['a12']), float(request.form['a13'])],
                [float(request.form['a21']), float(request.form['a22']), float(request.form['a23'])],
                [float(request.form['a31']), float(request.form['a32']), float(request.form['a33'])]
            ]
            b = [
                float(request.form['b1']),
                float(request.form['b2']),
                float(request.form['b3'])
            ]

            result, steps, status = gauss_seidel_method(A, b)

        except:
            error_message = "Invalid input. Please enter valid numeric values."

    return render_template("gauss.html", result=result, steps=steps, status=status, error=error_message)


# =======================
# TRAPEZOIDAL RULE
# =======================
@app.route('/trapezoidal', methods=['GET', 'POST'])
def trapezoidal():
    result = None
    table = None
    h = None
    error_message = None

    if request.method == 'POST':
        try:
            f_expr = request.form['function']
            a = float(request.form['a'])
            b = float(request.form['b'])
            n = int(request.form['n'])

            h = (b - a) / n
            table = []
            total = 0

            for i in range(n + 1):
                x = a + i * h
                fx = eval(f_expr)
                table.append((x, fx))

                if i == 0 or i == n:
                    total += fx
                else:
                    total += 2 * fx

            result = (h / 2) * total

        except:
            error_message = "Invalid input or function format."

    return render_template("trapezoidal.html", result=result, table=table, h=h, error=error_message)


# =======================
# MILNE METHOD
# =======================
@app.route('/milne', methods=['GET', 'POST'])
def milne():
    result = None
    steps = None
    error_message = None

    if request.method == 'POST':
        try:
            f_expr = request.form['function']
            x0 = float(request.form['x0'])
            y0 = float(request.form['y0'])
            h = float(request.form['h'])
            steps_count = int(request.form['steps'])

            def f(x, y):
                return eval(f_expr)

            values = [(x0, y0)]
            x = x0
            y = y0

            # First 3 values using Euler
            for _ in range(3):
                y = y + h * f(x, y)
                x = x + h
                values.append((x, y))

            # Milne Predictor-Corrector
            for i in range(3, steps_count):
                x4 = values[i][0] + h

                yp = values[i-3][1] + (4*h/3) * (
                    2*f(*values[i-2]) - f(*values[i-1]) + 2*f(*values[i])
                )

                yc = values[i-1][1] + (h/3) * (
                    f(*values[i-1]) + 4*f(*values[i]) + f(x4, yp)
                )

                values.append((x4, yc))

            result = values[-1]
            steps = values

        except:
            error_message = "Invalid input or function format."

    return render_template("milne.html", result=result, steps=steps, error=error_message)


if __name__ == '__main__':
    app.run(debug=True)