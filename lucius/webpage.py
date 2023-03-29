from dataclasses import dataclass


@dataclass
class Webpage:
    title: str
    inner_html: str

    @property
    def html(self):
        return f"""
<!DOCTYPE html>
<html color-mode="user">
    <head>
    <link rel="stylesheet" href="https://unpkg.com/mvp.css@1.12/mvp.css">
    <title>{self.title}</title>
</head>

<main>
    <section>
        <header>
            <h1>{self.title}</h1>
            <p>Made with ♥ by <a href="https://github.com/malyvsen/lucius">Lucius</a></p>
        </header>
        {self.inner_html}
    </section>

    </section>
</main>

</body>
</html>
        """.strip()
