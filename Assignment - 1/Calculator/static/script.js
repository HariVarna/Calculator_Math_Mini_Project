const examples = {
    gauss: {
        a11: 10, a12: -1, a13: 2,
        a21: -1, a22: 11, a23: -1,
        a31: 2, a32: -1, a33: 10,
        b1: 6, b2: 25, b3: -11
    },
    trapezoidal: {
        function: "x**2 + 1",
        a: 0,
        b: 4,
        n: 8
    },
    milne: {
        function: "x + y",
        x0: 0,
        y0: 1,
        h: 0.1,
        steps: 8
    }
};

function fillFormExample(form, key) {
    const data = examples[key];
    if (!data) return;

    Object.entries(data).forEach(([name, value]) => {
        const field = form.querySelector(`[name="${name}"]`);
        if (field) field.value = value;
    });
}

document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-fill-example]").forEach((btn) => {
        btn.addEventListener("click", () => {
            const key = btn.getAttribute("data-fill-example");
            const form = btn.closest("form");
            if (form) fillFormExample(form, key);
        });
    });

    document.querySelectorAll("form").forEach((form) => {
        form.addEventListener("submit", (event) => {
            const nField = form.querySelector("[name='n']");
            if (nField && nField.value && Number(nField.value) <= 0) {
                event.preventDefault();
                alert("n must be greater than zero.");
                return;
            }

            const stepsField = form.querySelector("[name='steps']");
            if (stepsField && stepsField.value && Number(stepsField.value) < 4) {
                event.preventDefault();
                alert("For Milne method, total points must be at least 4.");
            }
        });
    });
});
