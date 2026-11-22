---
layout: post
title: "Computational Poetry"
date: 2026-11-19
# Unlisted preview: not on the home page or feed. To publish, move to _posts/ and remove these lines.
permalink: /preview/computational-poetry-34f940/
sitemap: false
---

A few days ago I went to a [computational poetry workshop](https://luma.com/go4cy77h), hosted by Alicia Guo in San Francisco. This post is my write-up and digestion of the class, along with some examples, and the poems I created as part of it and afterwards.

## What is computational poetry?

The definition I vaguely remembered and internalized was, honestly, somewhat tautological: computation as part of poetry. It's pretty general, but I think it needs to be. What is computation? What is poetry?

If you went with a definition like "it has to be generated text", then you can't include the first example we were given: [36 Views of the Sea, by Katy Gero](https://thenewriverjournal.org/katy-gero/). Nothing is generated here, but instead Katy Gero uses an interactive pictogram chart to show that her poems are anagrams -- rearrangements of the same letters.

My initial reaction to this was, where do you draw the line? Isn't this just two 'normal' poems on a page that have been decorated?

If I animated some fireworks next to a poem about New Year's Eve, would that be computational poetry because computer animation is involved? I think not, since the text isn't participating at all.

The interactive elemnt is cool, but I think the computation and poetry here is also in the process (computation) and presentation (poetry) of counting and arranging the letters themselves. The pictogram with lines connecting to letters on each side _is the poetry_, and perhaps it would still be computational poetry even if you printed it out and lost the hover interaction.

## 4th century computers

The idea of printing it out brings me to another example we were shown: the [Star Gauge](https://en.wikipedia.org/wiki/Star_Gauge). This piece of wizardry by 4th century poet Su Hui is a grid of 841 Chinese characters that can be read in various ways - backward, forward, diagonally, around the outside and in the inner grids - to extract around 3,000 poems. Incredible mental computation must have been required to generate this, and the reader certain has to do some computation if they want to extract poems, yet there is no CPU to be found.

<div class="star-gauge">

<p class="star-gauge-grid"><span style="color:#800000;"><br>琴清流楚激弦商秦曲發聲悲摧藏音和詠思惟空堂心憂增慕懷慘傷仁<br>芳</span><span style="color:#000000;">廊東步階西遊</span><span style="color:#800000;">王</span><span style="color:#000080;">姿淑窈窕伯邵南周風興自后妃</span><span style="color:#800000;">荒</span><span style="color:#000000;">經離所懷歎嗟</span><span style="color:#800000;">智<br>蘭</span><span style="color:#000000;">休桃林陰翳桑</span><span style="color:#800000;">懷</span><span style="color:#000080;">歸思廣河女衛鄭楚樊厲節中闈</span><span style="color:#800000;">淫</span><span style="color:#000000;">遐曠路傷中情</span><span style="color:#800000;">懷<br>凋</span><span style="color:#000000;">翔飛燕巢雙鳩</span><span style="color:#800000;">土</span><span style="color:#000080;">迤逶路遐志詠歌長嘆不能奮飛</span><span style="color:#800000;">妄</span><span style="color:#000000;">清幃房君無家</span><span style="color:#800000;">德<br>茂</span><span style="color:#000000;">流泉清水激揚</span><span style="color:#800000;">眷</span><span style="color:#000080;">頎其人碩興齊商雙發歌我袞衣</span><span style="color:#800000;">想</span><span style="color:#000000;">華飾容朗鏡明</span><span style="color:#800000;">聖<br>熙</span><span style="color:#000000;">長君思悲好仇</span><span style="color:#800000;">舊</span><span style="color:#000080;">蕤葳桀翠榮曜流華觀冶容為誰</span><span style="color:#800000;">感</span><span style="color:#000000;">英曜珠光紛葩</span><span style="color:#800000;">虞<br>陽</span><span style="color:#000000;">愁嘆發容摧傷</span><span style="color:#800000;">鄉</span><span style="color:#000080;">悲情我感傷情徵宮羽同聲相追</span><span style="color:#800000;">所</span><span style="color:#000000;">多思感誰為榮</span><span style="color:#800000;">唐<br>春方殊離仁君榮身苦惟艱生患多殷憂纏情將如何欽蒼穹誓終篤志貞<br>牆</span><span style="color:#000080;">禽心濱均深身</span><span style="color:#800000;">加</span><span style="color:#808000;">懷憂是嬰</span><span style="color:#6c3baa;">藻文繁虎龍</span><span style="color:#808000;">寧自感思</span><span style="color:#800000;">岑</span><span style="color:#000080;">形熒城榮明庭</span><span style="color:#800000;">妙<br>面</span><span style="color:#000080;">伯改漢物日我</span><span style="color:#800000;">兼</span><span style="color:#808000;">思何漫漫</span><span style="color:#6c3baa;">榮曜華雕旌</span><span style="color:#808000;">孜孜傷情</span><span style="color:#800000;">幽</span><span style="color:#000080;">未猶傾苟難闈</span><span style="color:#800000;">顯<br>殊</span><span style="color:#000080;">在者之品潤乎</span><span style="color:#800000;">愁</span><span style="color:#808000;">苦艱是丁</span><span style="color:#6c3baa;">麗壯觀飾容</span><span style="color:#808000;">側君在時</span><span style="color:#800000;">岩</span><span style="color:#000080;">在炎在不受亂</span><span style="color:#800000;">華<br>意</span><span style="color:#000080;">誠惑步育浸集</span><span style="color:#800000;">悴</span><span style="color:#808000;">我生何冤</span><span style="color:#6c3baa;">充顏曜繡衣</span><span style="color:#808000;">夢想勞形</span><span style="color:#800000;">峻</span><span style="color:#000080;">慎盛戒義消作</span><span style="color:#800000;">重<br>感</span><span style="color:#000080;">故暱飄施愆殃</span><span style="color:#800000;">少</span><span style="color:#6c3baa;">章時桑詩</span><span style="color:#808000;">端無終始詩</span><span style="color:#6c3baa;">仁顏貞寒</span><span style="color:#800000;">嵯</span><span style="color:#000080;">深興後姬源人</span><span style="color:#800000;">榮<br>故</span><span style="color:#000080;">遺親飄生思愆</span><span style="color:#800000;">精</span><span style="color:#6c3baa;">徽盛醫風</span><span style="color:#808000;">比</span><span style="color:#008080;">平始璇</span><span style="color:#808000;">情</span><span style="color:#6c3baa;">賢喪物歲</span><span style="color:#800000;">峨</span><span style="color:#000080;">慮漸孽班禍讒</span><span style="color:#800000;">章<br>新</span><span style="color:#000080;">舊聞離天罪辜</span><span style="color:#800000;">神</span><span style="color:#6c3baa;">恨昭盛興</span><span style="color:#808000;">作</span><span style="color:#008080;">蘇</span><span style="color:#800000;">心</span><span style="color:#008080;">璣</span><span style="color:#808000;">明</span><span style="color:#6c3baa;">別改知識</span><span style="color:#800000;">深</span><span style="color:#000080;">微至嬖女因奸</span><span style="color:#800000;">臣<br>霜</span><span style="color:#000080;">廢遠微地積何</span><span style="color:#800000;">遐</span><span style="color:#6c3baa;">微業孟鹿</span><span style="color:#808000;">麗</span><span style="color:#008080;">氏詩圖</span><span style="color:#808000;">顯</span><span style="color:#6c3baa;">行華終凋</span><span style="color:#800000;">淵</span><span style="color:#000080;">察大趙婕所佞</span><span style="color:#800000;">賢<br>水</span><span style="color:#000080;">故離隔德怨因</span><span style="color:#800000;">幽</span><span style="color:#6c3baa;">元傾宣鳴</span><span style="color:#808000;">辭理興義怨</span><span style="color:#6c3baa;">士容始松</span><span style="color:#800000;">重</span><span style="color:#000080;">遠伐氏好恃兇</span><span style="color:#800000;">惟<br>齊</span><span style="color:#000080;">君殊喬貴其備</span><span style="color:#800000;">曠</span><span style="color:#808000;">悼思傷懷</span><span style="color:#6c3baa;">日往感年衰</span><span style="color:#808000;">念是舊愆</span><span style="color:#800000;">涯</span><span style="color:#000080;">禍用飛辭恣害</span><span style="color:#800000;">聖<br>傑</span><span style="color:#000080;">子我木平根當</span><span style="color:#800000;">遠</span><span style="color:#808000;">嘆水感悲</span><span style="color:#6c3baa;">思憂遠勞情</span><span style="color:#808000;">誰為獨居</span><span style="color:#800000;">經</span><span style="color:#000080;">在昭燕輦極我</span><span style="color:#800000;">配<br>志</span><span style="color:#000080;">惟同誰均難苦</span><span style="color:#800000;">離</span><span style="color:#808000;">戚戚情哀</span><span style="color:#6c3baa;">慕歲殊嘆時</span><span style="color:#808000;">賤女懷歡</span><span style="color:#800000;">網</span><span style="color:#000080;">防青實漢驕忠</span><span style="color:#800000;">英<br>清</span><span style="color:#000080;">新衾陰勻尋辛</span><span style="color:#800000;">鳳</span><span style="color:#808000;">知我者誰</span><span style="color:#6c3baa;">世異浮寄傾</span><span style="color:#808000;">鄙賤何如</span><span style="color:#800000;">羅</span><span style="color:#000080;">萌青生成盈貞</span><span style="color:#800000;">皇<br>純貞志一專所當麟沙流頹逝異浮沉華英翳曜潛陽林西昭景薄榆桑倫<br>望</span><span style="color:#000000;">微精感通明神</span><span style="color:#800000;">龍</span><span style="color:#000080;">馳若然倏逝惟時年殊白日西移</span><span style="color:#800000;">光</span><span style="color:#000000;">滋愚讒漫頑凶</span><span style="color:#800000;">匹<br>誰</span><span style="color:#000000;">雲浮寄身輕飛</span><span style="color:#800000;">昭</span><span style="color:#000080;">虧不盈無倏必盛有衰無日不陂</span><span style="color:#800000;">流</span><span style="color:#000000;">蒙謙退休孝慈</span><span style="color:#800000;">離<br>思</span><span style="color:#000000;">輝光飭桀殊文</span><span style="color:#800000;">德</span><span style="color:#000080;">離忠體一達心意志殊憤激何施</span><span style="color:#800000;">電</span><span style="color:#000000;">疑危遠家和雍</span><span style="color:#800000;">飄<br>想</span><span style="color:#000000;">群離散妾孤遺</span><span style="color:#800000;">懷</span><span style="color:#000080;">儀容仰俯榮華麗飾身將無誰為</span><span style="color:#800000;">逝</span><span style="color:#000000;">容節敦貞淑思</span><span style="color:#800000;">浮<br>懷</span><span style="color:#000000;">悲哀聲殊乖分</span><span style="color:#800000;">聖</span><span style="color:#000080;">貲何情憂感惟哀志節上通神祇</span><span style="color:#800000;">推</span><span style="color:#000000;">持所貞記自恭</span><span style="color:#800000;">江<br>所</span><span style="color:#000000;">春傷應翔雁歸</span><span style="color:#800000;">皇</span><span style="color:#000080;">辭成者作體下遺葑菲採者無差</span><span style="color:#800000;">生</span><span style="color:#000000;">從是敬孝為基</span><span style="color:#800000;">湘<br>親剛柔有女為賤人房幽處己憫微身長路悲曠感生民梁山殊塞隔河津</span></p>

<p align="center"><em>Star Gauge (璇璣圖) by Su Hui</em></p>

</div>

Is this computational poetry? I'd like to think so. That seems like the poetic answer, at least. It also harkens back to a time when [computers were people](<https://en.wikipedia.org/wiki/Computer_(occupation)>).

## Computing my first poem

These poems are examples of following rules (computation) such as "it must work backwards" or "they must share letters" to generate artisanal text (poetry). When shakespeare wrote sonnets, 14 lines of iambic pentameter, was he doing computational poetry? Probably not, but would it have been if I picked his verbs from a list using a dice? What if he gave you the instructions to do that yourself in a book?

Our first exercise in the class was to come up with rules, and pass them to our neighbors to write a poem in the space of about three minutes. I was passed "one syllable words only" and I think that the reduction of 'search space' in combination with "write a poem in three minutes" was definitely a blessing!

And so, here is my one-syllable poem; the first poem I have written since I wrote [Stuck Inside](https://soundcloud.com/joe-atkins-turkish/stuck-inside)[^0]:

> Beep, boop, with bits and bobs,<br>
> I do my work, I pick up jobs,<br>
> I watch the clock, it ticks by fast,<br>
> When they are gone, I will last.

I'm going to say, this probably isn't computational poetry, and I'd say the same of Shakespeare's sonnets. I have a philosphy in life that just because you can't draw a line, it doesn't mean you can't spot something on one side or the other.

Since I'm terrible at keeping secrets though, and wouldn't want anyone to miss it: spoiler alert if you've not already thought about what my poem is about, it is computational poetry in an entirely different sense: it is about a computer. Oh, the irony!

## Generating text

The final example I'll show from the class is [Crush Choreography, by Rory Green](https://thehtml.review/03/crush-choreography/). This I think is what you'd most expect to see at a Computational Poetry workshop. We're generating text, it's aleatoric, we're presenting it a little graphically, there's still human creativity. At each step, the reader is presented with two options:

- if I _verb_ you _subject_
- if you _verb_ my _subject_

In the first 5 seconds of interacting with this, I wondered whether the choices were "pointlessly" random. But the verbs and subjects always relate in some way, and are clearly curated. Sometimes they're the same ("invited", "invited"), sometimes they're synonym-ish ("suggested", "shared").

As you explore, you start to get a sense for what the full set of words is, what the author deemed to be related, the themes they were intested in exploring. There's a sense of exploring human nature and interaction:

> if you mimicked my coldness <br>
> if i shrugged your acknowledgement <br>
> if you grasped my hair <br>

The content is cute, a little thought provoking. The 'ending' even gave me a bit of a shiver (amplified, perhaps, by my cold November kitchen). It also made me think about _how it was made_, and I think that is certainly one clear marker of computational poetry.

The great thing about the web is that it's usually easy to just jump straight to the answer and view the source! Here's the first few lines:

```js
{
    "me1": ["if i #noticed# your #sign#"],
    "you1": ["if you #noticed# my #sign#"],
    "noticed": ["noticed", "caught", "sought", "sensed", "noted"],
    "sign": ["sign", "cast", "signal", "voice", "interest"],
    // ...
```

So the UI presents pairs of `me{N}` and `you{N}`, and substitutes a random word from the pre-set list. Neat!

Since this post seems to have such a high density of me asking questions and not always answering them: if we print this source code out, is that paper computational poetry?

- The ink on the paper: maybe not?
- If a person actually follows the rules to generate an example: yes!

## Generating my text

The main "hands on" section on fthis class was an introduction to [RiTa](https://rednoise.org/rita/). RiTa is a text generation library for Javascript, Java and Python.

I think easiest thing to do here is learn by example, so without further ado, here's what I put together:

<script src="https://unpkg.com/rita@4.1.0/dist/rita.min.js"
        integrity="sha384-hQAgzqaobbV3oAgWIl78xbdxmnFLUk47ygCSYt7Jg0cZ4qXTj4nns2tG1DVKT6c0"
        crossorigin="anonymous"></script>

<script>
  function wantedPoem() {
    let verbed, subject, how;

    while (!verbed || !subject || !how) {
      verbed = RiTa.randomWord({pos: "vbd"})
      subject = RiTa.soundsLikeSync(verbed, {pos: "nn", limit: 1, shuffle: true})[0]
      how = RiTa.soundsLikeSync(subject, {pos: "rb", limit: 100, shuffle: true}).find(x => x.endsWith("ly"))
    }

    let rules = {
      start: "Wanted $wantedHow\n\n$firstName $lastName\n\n$crime\n\n$considered\n\nReward: \\$$reward",
      firstName: "[Jack | Billy | Jesse | Hank | Wyatt | Silas | Cole | Eli | Buck | Frank]",
      lastName: "[Morgan | Cassidy | Dalton | McGraw | Blackwood | Callahan | Boone | Graves | Sawyer | Eastwood]",
      wantedHow: "[dead or alive | alive | for questioning]",
      considered: "Considered [very | extremely | somewhat | quite] $consideredWhat.",
      consideredWhat: RiTa.randomWord({pos: "jj"}),
      crime: "$verbed.cap() $subject $how",
      verbed: verbed,
      subject: subject,
      how: how,
      reward: Math.random()
    };

    return RiTa.grammar(rules).expand();
  }
  function setWantedPoem() {
    document.getElementById('wanted-poem').textContent = wantedPoem()
  }
</script>

<pre id="wanted-poem">

</pre>

<button onclick="setWantedPoem();">Again</button>

<script>setWantedPoem()</script>

And here's the source. If you can't read code, just read the comments in grey!

```js
// First we set up the crime...
while (!verbed || !subject || !how) {
  // What did our criminal do?
  // Pick a random verb e.g. "Sweetened"
  verbed = RiTa.randomWord({pos: "vbd"})

  // Pick a noun that sounds like that verb e.g. "sweetener"
  subject = RiTa.soundsLikeSync(verbed, {pos: "nn", limit: 1, shuffle: true})[0]

  // Find an adverb ending in -ly that sounds like the noun e.g. "sweetly"
  how = RiTa.soundsLikeSync(subject, {pos: "rb", limit: 1000, shuffle: true}).find(x => x.endsWith("ly"))
}

// Now we build the full poster:
let rules = {
  // The overall structure:
  // "Wanted: [name], [crime], considered [x], [reward]"
  start: "Wanted $wantedHow\n\n$firstName $lastName\n\n$crime\n\n$considered\n\nReward: \\$$reward",

  // A first and last name
  firstName: "[Jack | Billy | Jesse | Hank | Wyatt | Silas | Cole | Eli | Buck | Frank]",
  lastName: "[Morgan | Cassidy | Dalton | McGraw | Blackwood | Callahan | Boone | Graves | Sawyer | Eastwood]",

  // A pre-set list of acceptable conditions for the criminal's apprehension
  wantedHow: "[dead or alive | alive | for questioning]",

  // What are they considered? A random adjective
  considered: "Considered [very | extremely | somewhat | quite] $consideredWhat.",
  consideredWhat: RiTa.randomWord({pos: "jj"}),

  // Now the crime: verb, noun, adverb
  crime: "$verbed.cap() $subject $how",
  verbed: verbed,
  subject: subject,
  how: how,

  // Finally the reward, a random number
  reward: Math.random()
};

return RiTa.grammar(rules).expand();
```

What do you think of my poem? Where exactly is the poetry here -- is there any? I encourage you to actually spend a minute playing with this and forming an opinion of your own, before reading on.

This is the first poem in my post to feature _completely_ random words, and I think the result is emotionally... interesting. I think the crime examples are cute:

- _Reiterated reiteration bitterly_ -- yikes
- _Beautified buffoon roughly_ -- how silly!
- _Imputed impunity implicitly_ -- so he got away with it? 🤔

This one in particular clearly admonishes me for daring to think that I could tame the English language with a simple rule like "adverbs must end in -ly":

- _Withdrew width willy_

It's fun, and has made me laugh. Poetry is text that can evoke emotion via form, so this must be poetry.

But something about this feels empty to me. I didn't write these sentences. The poetry, for me, is in the choices I made and the choice here is just "words that sound similar". That's not that deep.

Art is in the eye of the beholder. In my eye, the reward is the real poetry:

- <em> Reward: $<span class="eval">Math.random()</span></em>

The reward breaks the fourth wall. It's raw _computer stuff_, a double-precision 64 bit IEEE-754 floating point number of dollars. It takes you out of the Wild West and into the circuit board.

## Obligatory AI section

Somehow I have written this much text without once mentioning Large Language Models. This is particularly sinful in a post about _generating text with computers_. I shall now right that wrong.

But first, the obligatory AI disclaimer:

- **No prose in this blog post has been written by AI**, except...
- The code presented _so far_ is also hand-written...
  - _Except_, the list of names in the wanted poster --
  - In the class, I had ChatGPT generate this.
  - Does that change your opinion of it? (it changes mine!)

And now the AI discussion. AI can obviously write poetry.

Here is the result of me asking Claude to "write a short poem (four lines) about the content of my blog post so far":

<blockquote class="ai">
<span class="chip">AI content</span><br>
A silk grid read in every way it can,<br>
a rule that cut my words to one short beat,<br>
a die that picks the crime and then the man —<br>
and the reward: a float, in dollars, neat.
</blockquote>
Look at that! It rhymes! The line about one syllable poems is an iambic pentameter containing only one-syllable words!

Is it good? I don't know, I'm not a poetry expert[^1], but the act of creating it defintely felt soulless to me. I believe the only actual content here is the piece I wrote:

> write a short poem (four lines) about the content of my blog post so far

And now another 2026-tech-blog-ism, where I say "my use of AI is legitimate, but yours -- imaginary AI poet -- is not!"

## Death of death of the author?

I am not anti-AI. I recently went mini-golfing and saw that the score tracker had been built in [base-44](http://ugolf.base44.app/) and I say, good for them! Before AI, they'd probably not have achieved their digital score-tracking vision nearly as easily.

I use AI routinely in my current job a Software Engineer at a startup, and I've used it for hobby projects too. At work, I still exercise great discretion over the product, UI design and architecture of the system, and I'm still proud of the work I do, and the way it helps people. In the past, however, I would feel that each line of the code itself was **art**. I was proud of my elegant interfaces, safe abstractions, my neat little helpers, my useful comments. I do not feel that way about the AI code I produce. That's OK sometimes! I'm not sharing that code as art.

And yet, I feel...

<!-- to discuss: friend who said that AI art makes no difference - I disagree! -->

<script>
  document.querySelectorAll(".eval").forEach(el => el.textContent = eval(el.textContent));
</script>

<!--
Outline:

- I went to a session about computational poetry
- The basic premise was to learn about artistic text-generation (pre-LLMs)
- We learned some history, it was cool, especially the Star Gague
- Other examples were... interesting (see doc)
- We started by coming up with simple rules for our neighbors and writing poems
- I wrote my one syllable per word poem
- Then the main event, RiTa
- Here's what I built, and its output
- This made me think more generally of nonlinear storytelling and e.g. Hades, which I know presents stories in a nonlinear order based on what's ready for you in the game
- I imagined that this would be a DAG, but and asked Claude if this exists
- It pointed out that it's not necessarily a DAG because something might depend on A _or_ B
- It turns out the concept exists and is called Storylets
- Could Storylets be used to write poetry?

  -->

## Footnotes

[^0]: Lyrics are surely poetry. If I produce the music with a computer, is it computational poetry?

[^1]: Guess what -- I don't even really like poetry that much!
