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

# Cards

> [!WARNING]
> Experimental, could change in the future or be removed

## Example

{{% columns %}}
- {{< card image="placeholder.svg" >}}
  ### Line 1
  Line 2
  {{< /card >}}

- {{< card image="placeholder.svg" >}}
  This is tab MacOS content.
  {{< /card >}}
{{% /columns %}}

{{% columns %}}
- {{< card href="/docs/shortcodes/experimental/cards" >}}
  **Markdown**
  Suspendisse sed congue orci, eu congue metus. Nullam feugiat urna massa.
  {{< /card >}}

- {{< card >}}
  Suspendisse sed congue orci, eu congue metus. Nullam feugiat urna massa, et fringilla metus consectetur molestie.
  {{< /card >}}

- {{< card title="Card" >}}
  ### Heading
  This is tab MacOS content.
  {{< /card >}}
{{% /columns %}}
