$(document).ready(function() {
    $('#upload-form').submit(function(event) {
        event.preventDefault();
        var formData = new FormData($(this)[0]);
        $.ajax({
            url: '/analyze',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $('#result-container').text('Accuracy: ' + response.accuracy);
            },
            error: function() {
                $('#result-container').text('Error occurred while analyzing the data.');
            }
        });
    });
});
