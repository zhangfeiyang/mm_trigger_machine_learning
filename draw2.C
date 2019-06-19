{

	gStyle->SetOptStat(0);
	
	ifstream fin("result");
	
	const int N = 200000;
	
	TH1F *h0 = new TH1F("h0","",200,-0.1,1.1);
	TH1F *h1 = new TH1F("h1","",200,-0.1,1.1);

	TH1F *h2 = new TH1F("h2","",120,60,180);
	TH1F *h3 = new TH1F("h3","",120,60,180);

	int P_real,PE;double P_fit;	
	
	int total_dark = 0;
	int detect_dark = 0;

	int total_physics = 0;
	int detect_physics = 0;
	for(int i=0;i<N;i++){
		fin>>P_real>>P_fit>>PE;
			
		if(P_real==0){
			total_dark++;
			if(PE<132) detect_dark++;
			h0->Fill(P_fit);
			h2->Fill(PE);
		}
		else{
			h1->Fill(P_fit);
			h3->Fill(PE);
			total_physics++;
			if(PE>=132) detect_physics++;
		}
	}
	
	cout << "rate is " << detect_dark*1.0/total_dark<<"\n";
	cout << "rate is " << detect_physics*1.0/total_physics<<"\n";
	
	h0->SetLineColor(kRed);
	h0->SetLineWidth(2);
	h1->SetLineWidth(2);
	h1->GetXaxis()->SetTitle("Prob of Physics");
	h1->GetYaxis()->SetTitle("Event");
	h1->GetYaxis()->SetTitleOffset(1);
	h1->Draw();
	h0->Draw("same");
	TLine *li = new TLine(0.895,0,0.895,24000);
	li->SetLineColor(kGreen);
	li->SetLineWidth(2);
	li->SetLineStyle(9);
	li->Draw();
	
	TLegend *le = new TLegend(0.4,0.7,0.6,0.9);
	le->AddEntry(h0,"Dark noise","l");
	le->AddEntry(h1,"Physics","l");
	le->Draw();
/*	
	
	h2->SetLineColor(kRed);
	h2->SetLineWidth(2);
	h3->SetLineWidth(2);
	h3->GetXaxis()->SetTitle("PE");
	h3->GetYaxis()->SetTitle("Event");
	h3->GetYaxis()->SetTitleOffset(1);
	h3->GetYaxis()->SetRangeUser(0,4000);
	h3->Draw();
	h2->Draw("same");
	TLine *li = new TLine(132,0,132,4000);
	li->SetLineColor(kGreen);
	li->SetLineWidth(2);
	li->SetLineStyle(9);
	li->Draw();
	
	TLegend *le = new TLegend(0.7,0.7,0.9,0.9);
	le->AddEntry(h2,"Dark noise","l");
	le->AddEntry(h3,"Physics","l");
	le->Draw();
*/
}
