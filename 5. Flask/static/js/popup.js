/*globals $:false */
$(function() {
	"use strict";
	$('#pop').on('click', function() {
		$('.imagepreview').attr('src', $('#imageresource').attr('src'));
		$('#imagemodal').modal('show');
		});
});