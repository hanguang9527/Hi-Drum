/*!
 * Start Bootstrap - Creative Bootstrap Theme (http://startbootstrap.com)
 * Code licensed under the Apache License v2.0.
 * For details, see http://www.apache.org/licenses/LICENSE-2.0.
 */

(function ($) {
    "use strict"; // Start of use strict

    // jQuery for page scrolling feature - requires jQuery Easing plugin
    $('a.page-scroll').bind('click', function (event) {
        var $anchor = $(this);
        $('html, body').stop().animate({
            scrollTop: ($($anchor.attr('href')).offset().top - 80)
        }, 1250, 'easeInOutExpo');
        event.preventDefault();
    });

    // Custom scrollspy logic
    function updateActiveNav() {
        var scrollPos = $(window).scrollTop() + 100; // offset for early trigger
        var abstractTop = $('#abstract').offset().top;
        var demoTop = $('#arr_demo').offset().top;

        $('.navbar-nav .page-scroll').parent().removeClass('active');

        if (scrollPos >= demoTop) {
            $('.navbar-nav .page-scroll[href="#arr_demo"]').parent().addClass('active');
        } else if (scrollPos >= abstractTop) {
            $('.navbar-nav .page-scroll[href="#abstract"]').parent().addClass('active');
        }
    }

    $(window).on('scroll resize', updateActiveNav);
    updateActiveNav(); // run once on load

    // Closes the Responsive Menu on Menu Item Click
    $('.navbar-collapse ul li a').click(function () {
        $('.navbar-toggle:visible').click();
    });

    // Fit Text Plugin for Main Header
    $("h1").fitText(
        1.2, {
        minFontSize: '35px',
        maxFontSize: '65px'
    }
    );

    // Offset for Main Navigation
    $('#mainNav').affix({
        offset: {
            top: 100
        }
    })

    // Initialize WOW.js Scrolling Animations
    new WOW().init();

})(jQuery); // End of use strict
