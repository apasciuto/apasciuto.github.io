/*======================================================
************   JAVASCRIPT   ************
======================================================*/

// MOBILE NAVBAR
// ============================================================
$(document).ready(function () {
  $(".navbar-toggle").on("click", function () {
	  $(this).toggleClass("active");
  });
  // RESET NAVBAR-TOGGLE WHEN WINDOWN RESIZES
  if (window.matchMedia('(min-width: 768px)').matches) {
    $(".navbar-toggle").removeClass("active");
  }
  $("button").on("click", function () {
    $(".navbar-toggle").removeClass("active");
  });
});

/* ========================================================================
 * Bootstrap: offcanvas.js v3.1.3
 * http://jasny.github.io/bootstrap/javascript/#offcanvas
 * ========================================================================
 * Copyright 2013-2014 Arnold Daniels
 *
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ======================================================================== */


+function ($) { "use strict";

  // OFFCANVAS PUBLIC CLASS DEFINITION
  // ============================================================
  var OffCanvas = function (element, options) {
    this.$element = $(element)
    this.options  = $.extend({}, OffCanvas.DEFAULTS, options)
    this.state    = null
    this.placement = null

    if (this.options.recalc) {
      this.calcClone()
      $(window).on('resize', $.proxy(this.recalc, this))
    }

    if (this.options.autohide)
      $(document).on('click', $.proxy(this.autohide, this))


    if (this.options.toggle) this.toggle()

    if (this.options.disablescrolling) {
        this.options.disableScrolling = this.options.disablescrolling
        delete this.options.disablescrolling
    }
  }

  OffCanvas.DEFAULTS = {
    toggle: true,
    placement: 'auto',
    autohide: true,
    recalc: true,
    disableScrolling: true,
    modal: false
  }

  OffCanvas.prototype.offset = function () {
    switch (this.placement) {
      case 'left':
      case 'right':  return this.$element.outerWidth()
      case 'top':
      case 'bottom': return this.$element.outerHeight()
    }
  }

  OffCanvas.prototype.calcPlacement = function () {
    if (this.options.placement !== 'auto') {
        this.placement = this.options.placement
        return
    }

    if (!this.$element.hasClass('in')) {
      this.$element.css('visiblity', 'hidden !important').addClass('in')
    }

    var horizontal = $(window).width() / this.$element.width()
    var vertical = $(window).height() / this.$element.height()

    var element = this.$element
    function ab(a, b) {
      if (element.css(b) === 'auto') return a
      if (element.css(a) === 'auto') return b

      var size_a = parseInt(element.css(a), 10)
      var size_b = parseInt(element.css(b), 10)

      return size_a > size_b ? b : a
    }

    this.placement = horizontal >= vertical ? ab('left', 'right') : ab('top', 'bottom')

    if (this.$element.css('visibility') === 'hidden !important') {
      this.$element.removeClass('in').css('visiblity', '')
    }
  }

  OffCanvas.prototype.opposite = function (placement) {
    switch (placement) {
      case 'top':    return 'bottom'
      case 'left':   return 'right'
      case 'bottom': return 'top'
      case 'right':  return 'left'
    }
  }

  OffCanvas.prototype.getCanvasElements = function() {
    // Return a set containing the canvas plus all fixed elements
    var canvas = this.options.canvas ? $(this.options.canvas) : this.$element

    var fixed_elements = canvas.find('*').filter(function() {
      return $(this).css('position') === 'fixed'
    }).not(this.options.exclude)

    return canvas.add(fixed_elements)
  }

  OffCanvas.prototype.slide = function (elements, offset, callback) {
    // Use jQuery animation if CSS transitions aren't supported
    if (!$.support.transition) {
      var anim = {}
      anim[this.placement] = "+=" + offset
      return elements.animate(anim, 350, callback)
    }

    var placement = this.placement
    var opposite = this.opposite(placement)

    elements.each(function() {
      if ($(this).css(placement) !== 'auto')
        $(this).css(placement, (parseInt($(this).css(placement), 10) || 0) + offset)

      if ($(this).css(opposite) !== 'auto')
        $(this).css(opposite, (parseInt($(this).css(opposite), 10) || 0) - offset)
    })

    this.$element
      .one($.support.transition.end, callback)
      .emulateTransitionEnd(350)
  }

  OffCanvas.prototype.disableScrolling = function() {
    var bodyWidth = $('body').width()
    var prop = 'padding-' + this.opposite(this.placement)

    if ($('body').data('offcanvas-style') === undefined) {
      $('body').data('offcanvas-style', $('body').attr('style') || '')
    }

    $('body').css('overflow', 'hidden')

    if ($('body').width() > bodyWidth) {
      var padding = parseInt($('body').css(prop), 10) + $('body').width() - bodyWidth

      setTimeout(function() {
        $('body').css(prop, padding)
      }, 1)
    }
    //disable scrolling on mobiles (they ignore overflow:hidden)
    $('body').on('touchmove.bs', function(e) {
      e.preventDefault();
    });
  }

  OffCanvas.prototype.enableScrolling = function() {
    $('body').off('touchmove.bs');
  }

  OffCanvas.prototype.show = function () {
    if (this.state) return

    var startEvent = $.Event('show.bs.offcanvas')
    this.$element.trigger(startEvent)
    if (startEvent.isDefaultPrevented()) return

    this.state = 'slide-in'
    this.calcPlacement();

    var elements = this.getCanvasElements()
    var placement = this.placement
    var opposite = this.opposite(placement)
    var offset = this.offset()

    if (elements.index(this.$element) !== -1) {
      $(this.$element).data('offcanvas-style', $(this.$element).attr('style') || '')
      this.$element.css(placement, -1 * offset)
      this.$element.css(placement); // Workaround: Need to get the CSS property for it to be applied before the next line of code
    }

    elements.addClass('canvas-sliding').each(function() {
      if ($(this).data('offcanvas-style') === undefined) $(this).data('offcanvas-style', $(this).attr('style') || '')
      if ($(this).css('position') === 'static') $(this).css('position', 'relative')
      if (($(this).css(placement) === 'auto' || $(this).css(placement) === '0px') &&
          ($(this).css(opposite) === 'auto' || $(this).css(opposite) === '0px')) {
        $(this).css(placement, 0)
      }
    })

    if (this.options.disableScrolling) this.disableScrolling()
    if (this.options.modal) this.toggleBackdrop()

    var complete = function () {
      if (this.state != 'slide-in') return

      this.state = 'slid'

      elements.removeClass('canvas-sliding').addClass('canvas-slid')
      this.$element.trigger('shown.bs.offcanvas')
    }

    setTimeout($.proxy(function() {
      this.$element.addClass('in')
      this.slide(elements, offset, $.proxy(complete, this))
    }, this), 1)
  }

  OffCanvas.prototype.hide = function (fast) {
    if (this.state !== 'slid') return

    var startEvent = $.Event('hide.bs.offcanvas')
    this.$element.trigger(startEvent)
    if (startEvent.isDefaultPrevented()) return

    this.state = 'slide-out'

    var elements = $('.canvas-slid')
    var placement = this.placement
    var offset = -1 * this.offset()

    var complete = function () {
      if (this.state != 'slide-out') return

      this.state = null
      this.placement = null

      this.$element.removeClass('in')

      elements.removeClass('canvas-sliding')
      elements.add(this.$element).add('body').each(function() {
        $(this).attr('style', $(this).data('offcanvas-style')).removeData('offcanvas-style')
      })

      this.$element.trigger('hidden.bs.offcanvas')
    }

    if (this.options.disableScrolling) this.enableScrolling()
    if (this.options.modal) this.toggleBackdrop()

    elements.removeClass('canvas-slid').addClass('canvas-sliding')

    setTimeout($.proxy(function() {
      this.slide(elements, offset, $.proxy(complete, this))
    }, this), 1)
  }

  OffCanvas.prototype.toggle = function () {
    if (this.state === 'slide-in' || this.state === 'slide-out') return
    this[this.state === 'slid' ? 'hide' : 'show']()
  }

  OffCanvas.prototype.toggleBackdrop = function (callback) {
    callback = callback || $.noop;
    if (this.state == 'slide-in') {
      var doAnimate = $.support.transition;

      this.$backdrop = $('<div class="modal-backdrop fade" />')
      .insertAfter(this.$element);

      if (doAnimate) this.$backdrop[0].offsetWidth // force reflow

      this.$backdrop.addClass('in')

      doAnimate ?
        this.$backdrop
        .one($.support.transition.end, callback)
        .emulateTransitionEnd(150) :
        callback()
    } else if (this.state == 'slide-out' && this.$backdrop) {
      this.$backdrop.removeClass('in');
      $('body').off('touchmove.bs');
      var self = this;
      if ($.support.transition) {
        this.$backdrop
          .one($.support.transition.end, function() {
            self.$backdrop.remove();
            callback()
            self.$backdrop = null;
          })
        .emulateTransitionEnd(150);
      } else {
        this.$backdrop.remove();
        this.$backdrop = null;
        callback();
      }
    } else if (callback) {
      callback()
    }
  }

  OffCanvas.prototype.calcClone = function() {
    this.$calcClone = this.$element.clone()
      .html('')
      .addClass('offcanvas-clone').removeClass('in')
      .appendTo($('body'))
  }

  OffCanvas.prototype.recalc = function () {
    if (this.$calcClone.css('display') === 'none' || (this.state !== 'slid' && this.state !== 'slide-in')) return

    this.state = null
    this.placement = null
    var elements = this.getCanvasElements()

    this.$element.removeClass('in')

    elements.removeClass('canvas-slid')
    elements.add(this.$element).add('body').each(function() {
      $(this).attr('style', $(this).data('offcanvas-style')).removeData('offcanvas-style')
    })
  }

  OffCanvas.prototype.autohide = function (e) {
    if ($(e.target).closest(this.$element).length === 0) this.hide()
  }

  // OFFCANVAS PLUGIN DEFINITION
  // ==========================
  var old = $.fn.offcanvas

  $.fn.offcanvas = function (option) {
    return this.each(function () {
      var $this   = $(this)
      var data    = $this.data('bs.offcanvas')
      var options = $.extend({}, OffCanvas.DEFAULTS, $this.data(), typeof option === 'object' && option)

      if (!data) $this.data('bs.offcanvas', (data = new OffCanvas(this, options)))
      if (typeof option === 'string') data[option]()
    })
  }

  $.fn.offcanvas.Constructor = OffCanvas

  // OFFCANVAS NO CONFLICT
  // ====================
  $.fn.offcanvas.noConflict = function () {
    $.fn.offcanvas = old
    return this
  }

  // OFFCANVAS DATA-API
  // =================
  $(document).on('click.bs.offcanvas.data-api', '[data-toggle=offcanvas]', function (e) {
    var $this   = $(this), href
    var target  = $this.attr('data-target')
        || e.preventDefault()
        || (href = $this.attr('href')) && href.replace(/.*(?=#[^\s]+$)/, '') //strip for ie7
    var $canvas = $(target)
    var data    = $canvas.data('bs.offcanvas')
    var option  = data ? 'toggle' : $this.data()

    e.stopPropagation()

    if (data) data.toggle()
      else $canvas.offcanvas(option)
  })

}(window.jQuery);

// CONTACT
// ============================================================
$(function(){
    document.getElementById("ss-submit").addEventListener("click", function(){
            alert("Your message has been sent!");
            // $( "#dialog-5" ).dialog( "close" );
        setTimeout(function() {
            document.getElementById("ss-form").reset();
            window.location = "{{ site.baseurl }}";
        }, 1000);
    })
});

// SEARCH
// ============================================================
// function(t) {
//     t.fn.simpleJekyllSearch = function(e) {
//         function n() {
//             u.keyup(function(e) {
//                 t(this).val().length ? i(r(t(this).val())) : o()
//             })
//         }
//         function r(e) {
//             var n = [];
//             return t.each(c, function(t, r) {
//                 for (var t = 0; t < a.length; t++)
//                     void 0 !== r[a[t]] && -1 !== r[a[t]].toLowerCase().indexOf(e.toLowerCase()) && (n.push(r),
//                     t = a.length)
//             }),
//             n
//         }
//         function i(e) {
//             o(),
//             l.append(t(s.searchResultsTitle)),
//             e.length ? t.each(e, function(e, n) {
//                 if (e < s.limit) {
//                     for (var r = s.template, e = 0; e < a.length; e++) {
//                         var i = new RegExp("{" + a[e] + "}","g");
//                         r = r.replace(i, n[a[e]])
//                     }
//                     l.append(t(r))
//                 }
//             }) : l.append(s.noResults)
//         }
//         function o() {
//             l.children().remove()
//         }
//         var s = t.extend({
//             jsonFile: "/search.json",
//             jsonFormat: "title,tags,url,date",
//             template: '<li><article><a href="{url}">{title} <span class="entry-date"><time datetime="{date}">{date}</time></span></a></article></li>',
//             searchResults: ".search-results",
//             searchResultsTitle: "<h4>Search results:</h4>",
//             limit: "10",
//             noResults: "<p>Oh snap!<br/><small>We found nothing :(</small></p>"
//         }, e)
//           , a = s.jsonFormat.split(",")
//           , c = []
//           , u = this
//           , l = t(s.searchResults);
//         s.jsonFile.length && l.length && t.ajax({
//             type: "GET",
//             url: s.jsonFile,
//             dataType: "json",
//             success: function(t, e, r) {
//                 c = t,
//                 n()
//             },
//             error: function(t, e, n) {
//                 console.log("***ERROR in simpleJekyllSearch.js***"),
//                 console.log(t),
//                 console.log(e),
//                 console.log(n)
//             }
//         })
//     }
// }(Zepto),
// function(t, e) {
//     "function" == typeof define && define.amd ? define([], e(t)) : "object" == typeof exports ? module.exports = e(t) : t.smoothScroll = e(t)
// }("undefined" != typeof global ? global : this.window || this.global, function(t) {
//     "use strict";
//     var e, n, r, i, o = {}, s = "querySelector"in document && "addEventListener"in t, a = {
//         selector: "[data-scroll]",
//         selectorHeader: "[data-scroll-header]",
//         speed: 500,
//         easing: "easeInOutCubic",
//         offset: 0,
//         updateURL: !0,
//         callback: function() {}
//     }, c = function() {
//         var t = {}
//           , e = !1
//           , n = 0
//           , r = arguments.length;
//         "[object Boolean]" === Object.prototype.toString.call(arguments[0]) && (e = arguments[0],
//         n++);
//         for (var i = function(n) {
//             for (var r in n)
//                 Object.prototype.hasOwnProperty.call(n, r) && (e && "[object Object]" === Object.prototype.toString.call(n[r]) ? t[r] = c(!0, t[r], n[r]) : t[r] = n[r])
//         }; r > n; n++) {
//             var o = arguments[n];
//             i(o)
//         }
//         return t
//     }, u = function(t) {
//         return Math.max(t.scrollHeight, t.offsetHeight, t.clientHeight)
//     }, l = function(t, e) {
//         var n, r, i = e.charAt(0), o = "classList"in document.documentElement;
//         for ("[" === i && (e = e.substr(1, e.length - 2),
//         n = e.split("="),
//         n.length > 1 && (r = !0,
//         n[1] = n[1].replace(/"/g, "").replace(/'/g, ""))); t && t !== document; t = t.parentNode) {
//             if ("." === i)
//                 if (o) {
//                     if (t.classList.contains(e.substr(1)))
//                         return t
//                 } else if (new RegExp("(^|\\s)" + e.substr(1) + "(\\s|$)").test(t.className))
//                     return t;
//             if ("#" === i && t.id === e.substr(1))
//                 return t;
//             if ("[" === i && t.hasAttribute(n[0])) {
//                 if (!r)
//                     return t;
//                 if (t.getAttribute(n[0]) === n[1])
//                     return t
//             }
//             if (t.tagName.toLowerCase() === e)
//                 return t
//         }
//         return null
//     }, f = function(t) {
//         for (var e, n = String(t), r = n.length, i = -1, o = "", s = n.charCodeAt(0); ++i < r; ) {
//             if (e = n.charCodeAt(i),
//             0 === e)
//                 throw new InvalidCharacterError("Invalid character: the input contains U+0000.");
//             o += e >= 1 && 31 >= e || 127 == e || 0 === i && e >= 48 && 57 >= e || 1 === i && e >= 48 && 57 >= e && 45 === s ? "\\" + e.toString(16) + " " : e >= 128 || 45 === e || 95 === e || e >= 48 && 57 >= e || e >= 65 && 90 >= e || e >= 97 && 122 >= e ? n.charAt(i) : "\\" + n.charAt(i)
//         }
//         return o
//     }, h = function(t, e) {
//         var n;
//         return "easeInQuad" === t && (n = e * e),
//         "easeOutQuad" === t && (n = e * (2 - e)),
//         "easeInOutQuad" === t && (n = .5 > e ? 2 * e * e : -1 + (4 - 2 * e) * e),
//         "easeInCubic" === t && (n = e * e * e),
//         "easeOutCubic" === t && (n = --e * e * e + 1),
//         "easeInOutCubic" === t && (n = .5 > e ? 4 * e * e * e : (e - 1) * (2 * e - 2) * (2 * e - 2) + 1),
//         "easeInQuart" === t && (n = e * e * e * e),
//         "easeOutQuart" === t && (n = 1 - --e * e * e * e),
//         "easeInOutQuart" === t && (n = .5 > e ? 8 * e * e * e * e : 1 - 8 * --e * e * e * e),
//         "easeInQuint" === t && (n = e * e * e * e * e),
//         "easeOutQuint" === t && (n = 1 + --e * e * e * e * e),
//         "easeInOutQuint" === t && (n = .5 > e ? 16 * e * e * e * e * e : 1 + 16 * --e * e * e * e * e),
//         n || e
//     }, p = function(t, e, n) {
//         var r = 0;
//         if (t.offsetParent)
//             do
//                 r += t.offsetTop,
//                 t = t.offsetParent;
//             while (t);return r = r - e - n,
//         r >= 0 ? r : 0
//     }, d = function() {
//         return Math.max(t.document.body.scrollHeight, t.document.documentElement.scrollHeight, t.document.body.offsetHeight, t.document.documentElement.offsetHeight, t.document.body.clientHeight, t.document.documentElement.clientHeight)
//     }, m = function(t) {
//         return t && "object" == typeof JSON && "function" == typeof JSON.parse ? JSON.parse(t) : {}
//     }, g = function(e, n) {
//         t.history.pushState && (n || "true" === n) && "file:" !== t.location.protocol && t.history.pushState(null, null, [t.location.protocol, "//", t.location.host, t.location.pathname, t.location.search, e].join(""));
//     }, v = function(t) {
//         return null === t ? 0 : u(t) + t.offsetTop
//     };
//     o.animateScroll = function(e, n, o) {
//         var s = m(e ? e.getAttribute("data-options") : null)
//           , u = c(u || a, o || {}, s);
//         n = "#" + f(n.substr(1));
//         var l = "#" === n ? t.document.documentElement : t.document.querySelector(n)
//           , y = t.pageYOffset;
//         r || (r = t.document.querySelector(u.selectorHeader)),
//         i || (i = v(r));
//         var b, w, x, E = p(l, i, parseInt(u.offset, 10)), S = E - y, C = d(), j = 0;
//         g(n, u.updateURL);
//         var T = function(r, i, o) {
//             var s = t.pageYOffset;
//             (r == i || s == i || t.innerHeight + s >= C) && (clearInterval(o),
//             l.focus(),
//             u.callback(e, n))
//         }
//           , O = function() {
//             j += 16,
//             w = j / parseInt(u.speed, 10),
//             w = w > 1 ? 1 : w,
//             x = y + S * h(u.easing, w),
//             t.scrollTo(0, Math.floor(x)),
//             T(x, E, b)
//         }
//           , N = function() {
//             b = setInterval(O, 16)
//         };
//         0 === t.pageYOffset && t.scrollTo(0, 0),
//         N()
//     }
//     ;
//     var y = function(t) {
//         var n = l(t.target, e.selector);
//         n && "a" === n.tagName.toLowerCase() && (t.preventDefault(),
//         o.animateScroll(n, n.hash, e))
//     }
//       , b = function(t) {
//         n || (n = setTimeout(function() {
//             n = null,
//             i = v(r)
//         }, 66))
//     };
//     return o.destroy = function() {
//         e && (t.document.removeEventListener("click", y, !1),
//         t.removeEventListener("resize", b, !1),
//         e = null,
//         n = null,
//         r = null,
//         i = null)
//     }
//     ,
//     o.init = function(n) {
//         s && (o.destroy(),
//         e = c(a, n || {}),
//         r = t.document.querySelector(e.selectorHeader),
//         i = v(r),
//         t.document.addEventListener("click", y, !1),
//         r && t.addEventListener("resize", b, !1))
//     }
//     ,
//     o
// }),
// function() {
//     for (var t = document.links, e = 0, n = t.length; n > e; e++)
//         t[e].hostname != window.location.hostname && (t[e].target = "_blank",
//         t[e].className += " externalLink")
// }(),
// function(t, e, n) {
//     function r() {
//         t(".search-wrapper").toggleClass("active"),
//         o.searchform.toggleClass("active"),
//         o.canvas.removeClass("search-overlay")
//     }
//     function i() {
//         var t, n = document.querySelector(".header-post .content");
//         t = e.scrollY,
//         500 >= t && null != n && (n.style.transform = "translateY(" + -t / 3 + "px)",
//         n.style.opacity = 1 - t / 500)
//     }
//     t("a#slide").click(function() {
//         t("#sidebar,a#slide,#fade").addClass("slide"),
//         t("#open").hide(),
//         t("#search").hide(),
//         t("#close").show()
//     }),
//     t("#fade").click(function() {
//         t("#sidebar,a#slide,#fade").removeClass("slide"),
//         t("#open").show(),
//         t("#search").show(),
//         t("#close").hide()
//     });
//     var o = {
//         close: t(".icon-remove-sign"),
//         searchform: t(".search-form"),
//         canvas: t("body"),
//         dothis: t(".dosearch")
//     };
//     o.dothis.on("click", function() {
//         t(".search-wrapper").toggleClass("active"),
//         o.searchform.toggleClass("active"),
//         o.searchform.find("input").focus(),
//         o.canvas.toggleClass("search-overlay"),
//         t(".search-field").simpleJekyllSearch()
//     }),
//     o.close.on("click", r),
//     document.addEventListener("keyup", function(e) {
//         27 == e.keyCode && t(".search-overlay").length && r()
//     }),
//     document.getElementsByClassName("home").length >= 1 && new AnimOnScroll(document.getElementById("grid"),{
//         minDuration: .4,
//         maxDuration: .7,
//         viewportFactor: .2
//     }),
//     smoothScroll.init({
//         selectorHeader: ".bar-header",
//         speed: 500,
//         updateURL: !1
//     }),
//     screen.width > 1024 && e.addEventListener("scroll", i)
// }(Zepto, window);