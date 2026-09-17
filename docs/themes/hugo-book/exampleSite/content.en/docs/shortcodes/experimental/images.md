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

# Images

> [!WARNING]
> Experimental, could change in the future or be removed

Image shortcode produces an image that can be clicked to expand.

## Example

```go-html-template
{{</* image src="placeholder.svg" alt="A placeholder" title="A placeholder" loading="lazy" */>}}
```
{{< image src="placeholder.svg" alt="A placeholder" title="A placeholder" loading="lazy" >}}

## Parameters

`src` {{< badge style="warning" title="Required" >}}
: The link to the image

`class` {{< badge style="info" title="Optional" >}}
: An optional CSS class name that will be applied to the `img` element

`alt` {{< badge style="info" title="Optional" >}}
: An optional alternate text for the image

`title` {{< badge style="info" title="Optional" >}}
: An optional title for the image

`loading` {{< badge style="info" title="Optional" >}}
: Sets `loading` control for the image: `lazy`, `eager` or `auto`
