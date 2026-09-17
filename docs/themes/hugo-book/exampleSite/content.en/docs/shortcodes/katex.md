---
title: KaTeX
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->


# KaTeX

KaTeX shortcode let you render math typesetting in markdown document. See [KaTeX](https://katex.org/)

{{% hint info %}}
**Override KaTeX initialization config**
To override the [initialization config](https://katex.org/docs/options.html) for KaTeX,
create a `katex.json` file in your `assets` folder!
{{% /hint %}}

# Example
{{< katex />}}


## Activation
KaTeX is activated on the page by first use of the shortcode or render block. you can force activation with empty `{{</* katex /*/>}}` and use delimiters defined in configuration in `assets/katex.json`.

## Rendering as block

{{% columns %}}

```latex
{{</* katex display=true >}}
f(x) = \int_{-\infty}^\infty\hat f(\xi)\,e^{2 \pi i \xi x}\,d\xi
{{< /katex */>}}
```

````latex
```katex
f(x) = \int_{-\infty}^\infty\hat f(\xi)\,e^{2 \pi i \xi x}\,d\xi
```
````

````latex
$$
f(x) = \int_{-\infty}^\infty\hat f(\xi)\,e^{2 \pi i \xi x}\,d\xi
$$
````

<--->

{{< katex display=true >}}
f(x) = \int_{-\infty}^\infty\hat f(\xi)\,e^{2 \pi i \xi x}\,d\xi
{{< /katex >}}

---

```katex
f(x) = \int_{-\infty}^\infty\hat f(\xi)\,e^{2 \pi i \xi x}\,d\xi
```

---

$$
f(x) = \int_{-\infty}^\infty\hat f(\xi)\,e^{2 \pi i \xi x}\,d\xi
$$

{{% /columns %}}

## Rendering inline
When KaTeX is active on the page it is possible to write inline expressions.

| Code | Output |
| --   | --     |
| `{{</* katex >}}\pi(x){{< /katex */>}}` | {{< katex >}}\pi(x){{< /katex >}} |
| `\\( \pi(x) \\)` | \\( \pi(x) \\) |

## Configuration
KaTeX configuration could be adjusted by editing `assets/katex.json` file. For example to enabled inline delimiters `$..$` put content below into the file.

```json
{
  "delimiters": [
    {"left": "$$", "right": "$$", "display": true},
    {"left": "$", "right": "$", "display": false},
    {"left": "\\(", "right": "\\)", "display": false},
    {"left": "\\[", "right": "\\]", "display": true}
  ]
}
```
