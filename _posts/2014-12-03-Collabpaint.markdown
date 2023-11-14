---
layout: post
title: "I hosted CollabPaint! And got annoyed at 4-years-ago-Joe."
date: 2014-12-03 20:00:00 -0700
categories: tech
---

I put my old collaborative whiteboard project [CollabPaint](http://paint.veryjoe.com) back up on the internet! Multiple people can log in to a room and paint at the same time, without having to register for an account or sign up.

<div class="thumbnailed">
  <a href="/images/collabpaint.png">
    <img src="/thumbnails/collabpaint.png" alt="Screenshot of Collabpaint" />
  </a>
</div>

You can set the size/colour/opacity for your brush and also draw lines, rectangles and ovals. There's even a chat box and the option to save download the thing you draw.

I wrote CollabPaint about four years ago, and it was one of my first forays in to node.js and socket.io after CollabBox. Like all of my other projects, I never really polished it. It's still missing a few things such as an undo button (and some overdue bugfixes...), but otherwise it gets the job done!

### On future-proofing Code and following best practice

Sadly I had trouble getting CollabBox to run, because I used old crusty versions of all the libraries and never made a package.json file to record which versions of what things I used, so everything is broken. But, [here](https://github.com/Spacerat/Collab-Box) it is on github anyway.

If I could go back four years, I'd tell myself the following set of related things:

- Keep a record of which versions of everything you use, if you can
- When you start using a new language or ecosystem, make sure you actually look up the best practices for the things you're using.
- For example, with node.js and npm, actually bother to write package.json files to track all your dependancies!

This kind of stuff should really go without saying, but it strikes me as interesting that - as a hobbyist coder four+ years ago - it didn't really occur to me. So if you ever see your hobby-programmer kids writing code and they forget to make a package.json file, set them straight so they don't make the same mistakes I did!
