# 全民瘋AI系列2.0
> 第13屆iT邦幫忙鐵人賽

<iframe width="560" height="315" src="https://www.youtube.com/embed/C9mvGMtrPXo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## 前言
哈囉大家好我是10程式中的10！我是[上一屆](https://ithelp.ithome.com.tw/users/20107247/ironman/3719)鐵人賽影片教學組`全民瘋AI系列`的作者，當時講解了人工智慧的基礎以及常見的機器學習演算法與手把手教學。由於大家反應很熱烈，讓我看到了大家對於AI的學習熱忱。也因為上一屆獲得了影片教學組優選，收到了許多書商的出版邀請，由於我沒有時間與動力將這些大量知識寫成文章因此都婉拒了。因此我想藉由這一次鐵人賽將上一屆的影片內容整理成電子書版本，提供大家影片教學與文字版的筆記內容(唷呼書商快看過來～)當然內容會以之前影片教學為基底，並加入一些新的元素讓文章內容變得更紮實。在全新的`全民瘋AI系列2.0`中我會介紹實用的機器學習演算法並含有程式手把手實作，以及近年來熱門的機器學習套件與模型調參技巧。除此之外我還會提到大家最感興趣的 AI 模型落地與整合。希望在這次的鐵人賽能夠將AI的資源整理得更詳細並分享給各位。


![](./1.全民瘋AI系列2.0目標介紹/image/img1-1.jpg)

## 全民瘋AI系列電子書
全民瘋AI系列 是一個專為 AI 學習資源打造的開源平台，由一群熱愛資料科學的工程師所創立。這個平台的宗旨是提供一個開放、協作的環境，讓更多人能夠方便地學習 AI 和機器學習相關技術，無論是初學者還是進階使用者，都可以在這裡找到適合的學習資源和工具。透過社群的力量，平台上的內容持續更新，涵蓋從基礎理論到實務應用，滿足不同層次的學習需求。

| 書名                            | 簡介                              | 完成進度  | 討論區連結 |
|---------------------------------|-----------------------------------|-----------|------------|
| [Python從零開始](https://andy6804tw.github.io/crazyai-python/)| 適合初學者，詳細介紹Python語言的基本概念與程式設計技巧。 | 30%      | [加入討論](https://github.com/andy6804tw/crazyai-python/issues) |
| [經典機器學習](https://andy6804tw.github.io/crazyai-ml/)| 涵蓋各種經典的機器學習模型與演算法，從理論到實踐。         | 100%       | [加入討論](https://github.com/andy6804tw/crazyai-ml/issues) |
| [探索可解釋人工智慧](https://andy6804tw.github.io/crazyai-xai/)| 介紹解釋AI模型的最新技術與方法，幫助讀者理解AI決策的背後原因。 | 100%       | [加入討論](https://github.com/andy6804tw/crazyai-xai/issues) |
| [深度學習與神經網路](https://andy6804tw.github.io/crazyai-dl/)| 深入介紹深度學習與神經網路的概念與實作，適合進階讀者。       | 20%       | [加入討論](https://github.com/andy6804tw/crazyai-dl/issues) |
| [深度強化學習](https://andy6804tw.github.io/crazyai-rl/)| 涵蓋深度強化學習的理論與應用，適合對最佳化有深入興趣的讀者。   | 10%       | [加入討論](https://github.com/andy6804tw/crazyai-rl/issues) |

## 此系列教學適合誰?
如果您是之前的舊讀者，歡迎回來為自己充電～新的系列文章保證讓你收穫滿滿！若您是新來的讀者歡迎加入人工智慧的世界，此系列文章正適合初學者閱讀。另外建議可以搭配我[上一屆](https://ithelp.ithome.com.tw/users/20107247/ironman/3719)鐵人賽的影片教學進行學習。

## 系列文章內容規劃
在本次鐵人賽預計新增了許多新內容，特別是近年來比較新的演算法套件，以及在模型訓練中必須注意的大小事。本系列要在短短30天內講完所有 AI 領域相關應用是不太可能的事情，因此我的規劃是從認識人工智慧開始切入主題。先讓大家知道何謂人工智慧以及相關應用有哪些。接著帶各位了解成為資料科學家的第一步，就是資料分析與視覺化，再來會有一系列經典的機器學習演算法介紹。最後也是大家可能會有興趣的整合部分，會以實際的帶大家手把手部署我們的AI模型以及前後端串接的概念。


## 前置作業資源
本系列教學將有大量的程式實作，並採用 Google Colab 做為程式雲端運行的編輯執行環境。各位可以直接利用 Colab 開啟本系列文章的範例程式。在使用此平台之前每個人都必須要有自己的 Google 帳號，才能順利的開啟並執行程式碼。Colab 可讓你輕鬆地在瀏覽器上撰寫並執行 Python 程式語言，它可以說是機器學習新手的入門工具。此外 Colab 具備了以下幾個優點：

- 不必進行任何設定與安裝
- 免費額度使用 GPU、TPU 資源
- 輕鬆共用與分享檔案

因此讀者必須先熟悉 Colab 的操作模式，想了解該如何操作的朋友們可以先來看這一步[影片](https://youtu.be/C9mvGMtrPXo?t=266)教學。或是可以閱讀其他相關[文章](https://datasciocean.tech/python-tutorial/google-colaboratory/)。


## 回報錯誤與建議
本系列文章若有問題或是內容建議都可以來 GitHub 中的 [issue](https://github.com/andy6804tw/2021-13th-ironman/issues) 提出。歡迎大家一同貢獻為這系列文章有更好的閱讀品質。

## 關於作者
曾任職於台灣人工智慧學校，擔任AI工程師，擁有豐富的教學經驗，熱衷於網頁前後端整合與AI演算法的開發。希望藉由鐵人賽，將所學貢獻出來，為AI領域提供更多資源。

[@andy6804tw](https://github.com/andy6804tw)

歡迎大家訂閱我的 [YouTube](https://www.youtube.com/channel/UCSNPCGvMYEV-yIXAVt3FA5A) 頻道。

本系列教學簡報 PDF & Code 都可以從我的 [GitHub](https://github.com/andy6804tw/2021-13th-ironman) 取得！