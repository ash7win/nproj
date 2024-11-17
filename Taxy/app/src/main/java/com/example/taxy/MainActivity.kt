package com.example.taxy

import android.annotation.SuppressLint
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.widget.EditText
import android.widget.SeekBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.vectordrawable.graphics.drawable.ArgbEvaluator

private const val TAG ="MainActivity"
private const val INITIAL_TAX_PERCENT = 5
class MainActivity : AppCompatActivity() {
    private lateinit var etPrincipalAmount: EditText
    private lateinit var seekBarTax: SeekBar
    private lateinit var tvTaxPercentLabel: TextView
    private lateinit var tvTaxAmount: TextView
    private lateinit var tvTotalAmount: TextView
    private lateinit var tvTaxDescription: TextView

    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        etPrincipalAmount = findViewById(R.id.etPrincipalAmount)
        seekBarTax= findViewById(R.id.seekBarTax)
        tvTaxAmount = findViewById(R.id.tvTaxAmount)
        tvTotalAmount = findViewById(R.id.tvTotalAmount)
        tvTaxPercentLabel = findViewById(R.id.tvTaxPercentLabel)
        tvTaxDescription= findViewById(R.id.tvTaxDescription)

        seekBarTax.progress = INITIAL_TAX_PERCENT
        tvTaxPercentLabel.text="$INITIAL_TAX_PERCENT%"
        updateTaxDescription(INITIAL_TAX_PERCENT)
        seekBarTax.setOnSeekBarChangeListener(object: SeekBar.OnSeekBarChangeListener{
            @SuppressLint("SetTextI18n")
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                Log.i(TAG, "onProgressChanged $progress")
                tvTaxPercentLabel.text="$progress%"
                computeTaxAndTotal()
                updateTaxDescription(progress)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {
            }

            override fun onStopTrackingTouch(seekBar: SeekBar?) {
            }

        })
        etPrincipalAmount.addTextChangedListener(object:TextWatcher{
            override fun beforeTextChanged(p0: CharSequence?, p1: Int, p2: Int, p3: Int) {

            }

            override fun onTextChanged(p0: CharSequence?, p1: Int, p2: Int, p3: Int) {
            }

            override fun afterTextChanged(p0: Editable?) {
                Log.i(TAG, "afterTextChanged $p0")
                computeTaxAndTotal()
            }

        })
    }

    @SuppressLint("RestrictedApi")
    private fun updateTaxDescription(taxPercent: Int) {
        val taxDescription = when (taxPercent){
            in 0..9 -> "Great"
            in 10..14 -> "Acceptable"
            in 14..20 -> "Fine"
            in 20..35 -> "That's just robbery!"
            else -> "Don't pay this much tax"
        }
        tvTaxDescription.text = taxDescription
        val color =ArgbEvaluator().evaluate(
            taxPercent.toFloat()/seekBarTax.max,
            ContextCompat.getColor(this,R.color.best_tax),
            ContextCompat.getColor(this,R.color.worst_tax)
        ) as Int
        tvTaxDescription.setTextColor(color)
    }

    @SuppressLint("SetTextI18n")
    private fun computeTaxAndTotal() {
        if (etPrincipalAmount.text.isEmpty()){
            tvTotalAmount.text= ""
            tvTaxAmount.text= ""
            return
        }
        val principalAmount= etPrincipalAmount.text.toString().toDouble()
        val taxPercent = seekBarTax.progress
        val taxAmount = principalAmount * taxPercent /100
        val totalAmount = principalAmount + taxAmount
        tvTaxAmount.text = "%.2f".format(taxAmount)
        tvTotalAmount.text = "%.2f".format(totalAmount)
    }
}