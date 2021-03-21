from flask import Flask, render_template, request
from spam_detection import Filter

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/filter', methods=['POST'])
def filter():
    text = request.form.get('text')
    spam_filter = Filter()
    processed_input = spam_filter.pre_process(text)
    results, ham_count, spam_count = spam_filter.predict(processed_input)
    print('***********************************',results)
    
    return render_template(
        'results.html', 
        results=results, 
        ham_count=ham_count,
        spam_count=spam_count
    )

if __name__ == '__main__':
    app.run(debug=True)