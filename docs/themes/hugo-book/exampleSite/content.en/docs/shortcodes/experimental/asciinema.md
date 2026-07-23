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

# Asciinema

> [!WARNING]
> Experimental, could change in the future or be removed

Asciinema shortcode integrates asciinema player into the markdown page.

```tpl
{{</* asciinema
  cast="asciinema-627097.cast"
  or
  cast="https://asciinema.org/a/vJNKUQFjuh7qKI2j3OoaKs8Jk.cast"
  loop=true
  autoplay=true
  speed=2 */>}}
```

{{< asciinema
  cast="asciinema-627097.cast"
  loop=true
  autoplay=true
  speed=2 >}}

## Parameters

All parameters added to the shortcode will be transformed to options for Asciinema player, expect `cast` parameter that is used to locate cast file. Cast file follows same rules as portable image, it could be site resource, page resource or remote file URL.

[List of Asciinema options](https://docs.asciinema.org/manual/player/options/)
