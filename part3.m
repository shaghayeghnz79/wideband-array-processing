%% Part 2.3 
clear; close all; clc;

%% ===================== USER SWITCHES =====================
SAVE_AUDIO_FILES = true;

%% ===================== PRINT HEADER =====================
fprintf('=== PART 2.3 INTERFERENCE CANCELLATION ===\n\n');

%% ===================== LOAD DESIRED AUDIO =====================
desired_filename = "Ryan Gosling - City Of Stars (320).mp3";
if exist(desired_filename, 'file') ~= 2
    fprintf('Desired audio not found. Generating test desired signal...\n');
    fs = 44100; duration = 4;
    t_vec = (0:duration*fs-1)'/fs;
    s_desired = 0.5*sin(2*pi*440*t_vec).*(1+0.2*sin(2*pi*5*t_vec)) + ...
                0.3*sin(2*pi*880*t_vec).*(1+0.1*sin(2*pi*3*t_vec));
else
    [s_raw, fs] = audioread(desired_filename);
    if size(s_raw,2)>1, s_desired = mean(s_raw,2); else, s_desired = s_raw; end
    fprintf('Loaded desired: %s\n', desired_filename);
end

%% ===================== LOAD INTERFERENCE AUDIO =====================
interf_filename = "Lamb Of God – Walk With Me In Hell.mp3";
if exist(interf_filename, 'file') ~= 2
    fprintf('Interference audio not found. Generating test interference...\n');
    if ~exist('fs','var'), fs = 44100; end
    duration = 4;
    t_vec = (0:duration*fs-1)'/fs;
    s_interf = 0.4*(sin(2*pi*300*t_vec)+0.6*sin(2*pi*600*t_vec)+0.4*sin(2*pi*900*t_vec)) ...
               .*(1+0.3*sin(2*pi*8*t_vec));
else
    [i_raw, fs_i] = audioread(interf_filename);
    if fs_i ~= fs
        i_raw = resample(i_raw, fs, fs_i);
    end
    if size(i_raw,2)>1, s_interf = mean(i_raw,2); else, s_interf = i_raw; end
    fprintf('Loaded interference: %s\n', interf_filename);
end

%% ===================== TRIM / NORMALIZE =====================
minL = min(length(s_desired), length(s_interf));
s_desired = s_desired(1:minL);
s_interf  = s_interf(1:minL);

MAX_SEC = 4;
if length(s_desired) > MAX_SEC*fs
    s_desired = s_desired(1:floor(MAX_SEC*fs));
    s_interf  = s_interf(1:floor(MAX_SEC*fs));
end

t_vec = (0:length(s_desired)-1)'/fs;
Ns = length(t_vec);

eps0 = 1e-12;
s_desired = 0.7 * s_desired / (max(abs(s_desired))+eps0);
s_interf  = 0.5 * s_interf  / (max(abs(s_interf))+eps0);

fprintf('Signals ready. Duration: %.2f s | fs=%d Hz\n\n', Ns/fs, fs);

%% ===================== GEOMETRY + SOURCES =====================
c = 343;
N = 15;
d = 0.18;
L = 2.0;
omega0 = pi/2;
r0 = 1;

mic_x  = -(N-1)*d/2 + (0:N-1)*d;
p_mics = [mic_x; zeros(1,N)];

center_mic = ceil(N/2);           % 8
unaffected_mics = 1:center_mic;   % protected half
affected_mics   = (center_mic+1):N;

fprintf('Array: N=%d, d=%.3f m, center mic=%d, affected=%d..%d\n\n', ...
    N, d, center_mic, affected_mics(1), affected_mics(end));

% Desired rotating source (ellipse)
theta_t  = omega0 * t_vec;                         % Ns x 1
p_source = [2*L*cos(theta_t)'; L*sin(theta_t)'];   % 2 x Ns

%% ===================== PROPAGATE DESIRED =====================
fprintf('Simulating desired propagation...\n');
r_clean = zeros(Ns, N);
for k = 1:N
    dist_vec = sqrt(sum((p_source - p_mics(:,k)).^2, 1))';
    tau = dist_vec / c;
    s_del = interp1(t_vec, s_desired, t_vec - tau, 'linear', 0);
    gamma = (r0 ./ max(dist_vec,0.1)).^2;
    r_clean(:,k) = s_del .* gamma;
end

%% ===================== ADD STATIC INTERFERER =====================
fprintf('Adding static interferer to affected mics...\n');
interf_pos = [-2*L; 2*L];
interf_power_ratio = 0.8;

r_with_interf = r_clean;

for k = affected_mics
    dist_i  = norm(interf_pos - p_mics(:,k));
    tau_i   = dist_i / c;
    i_del   = interp1(t_vec, s_interf, t_vec - tau_i, 'linear', 0);
    gamma_i = (r0 / max(dist_i,0.1))^2;

    P_des = mean(r_clean(:,k).^2);
    P_int = mean((i_del*gamma_i).^2);
    if P_int > 0
        sc = sqrt(P_des / P_int * interf_power_ratio);
    else
        sc = sqrt(interf_power_ratio);
    end

    r_with_interf(:,k) = r_with_interf(:,k) + sc * gamma_i * i_del;
end

%% ===================== ADD SENSOR NOISE =====================
snr_db = 25;
sig_power    = mean(var(r_with_interf));
noise_power  = sig_power / (10^(snr_db/10));
noise_matrix = sqrt(noise_power) * randn(size(r_with_interf));

r_matrix = r_with_interf + noise_matrix;
fprintf('Done. Sensor-domain SNR = %d dB\n\n', snr_db);

% Processing signal
r_proc = r_matrix;

%% ===================== WINDOWED DToA ESTIMATION =====================
fprintf('Estimating DToA (windowed xcorr)...\n');

win_len = round(0.10*fs);
hop     = round(0.50*win_len);
num_w   = floor((Ns-win_len)/hop) + 1;

max_geom_delay = ((N-1)*d/2)/c;
max_lag = ceil(1.5*max_geom_delay*fs);

estimated_delays = zeros(N, num_w);
wH = hann(win_len);

for wi = 1:num_w
    a = (wi-1)*hop + 1;
    b = a + win_len - 1;

    ref = (r_proc(a:b, center_mic) - mean(r_proc(a:b, center_mic))) .* wH;

    for k = 1:N
        if k == center_mic
            estimated_delays(k,wi) = 0;
            continue;
        end

        x = (r_proc(a:b,k) - mean(r_proc(a:b,k))) .* wH;
        [xc,lags] = xcorr(x, ref, max_lag, 'coeff');

        % ABS peak pick + ABS parabolic refinement (consistent)
        [~,pi] = max(abs(xc));
        lag = lags(pi);

        if pi>1 && pi<length(xc)
            y1 = abs(xc(pi-1)); y2 = abs(xc(pi)); y3 = abs(xc(pi+1));
            den = (y1 - 2*y2 + y3);
            if abs(den) > 1e-12
                lag = lag + (y1 - y3)/(2*den);
            end
        end

        estimated_delays(k,wi) = lag/fs;
    end
end

% Smooth delay tracks
for k = 1:N
    if k ~= center_mic
        estimated_delays(k,:) = movmean(estimated_delays(k,:), 5);
    end
end

% Interpolate to sample grid
centers = (0:num_w-1)'*hop + win_len/2;
centers = max(1,min(Ns,centers));

delays_smooth = zeros(Ns,N);
for k = 1:N
    delays_smooth(:,k) = interp1(centers, estimated_delays(k,:)', (1:Ns)', 'pchip', 'extrap');
end
delays_smooth(:,center_mic) = 0;

%% ===================== ALIGN SIGNALS (AUTO SIGN) =====================
fprintf('Aligning signals (auto sign selection)...\n');

aligned_plus  = zeros(Ns,N);
aligned_minus = zeros(Ns,N);

for k = 1:N
    aligned_plus(:,k)  = interp1(t_vec, r_proc(:,k), t_vec + delays_smooth(:,k), 'linear', 0);
    aligned_minus(:,k) = interp1(t_vec, r_proc(:,k), t_vec - delays_smooth(:,k), 'linear', 0);
end

valid_idx = max(1,floor(0.3*Ns)) : min(Ns,floor(0.7*Ns));

corr_plus  = 0;
corr_minus = 0;
for k = 1:N
    if k==center_mic, continue; end
    corr_plus  = corr_plus  + abs(corr(aligned_plus(valid_idx,k),  aligned_plus(valid_idx,center_mic)));
    corr_minus = corr_minus + abs(corr(aligned_minus(valid_idx,k), aligned_minus(valid_idx,center_mic)));
end

if corr_plus >= corr_minus
    aligned = aligned_plus;
    align_sign = +1;
else
    aligned = aligned_minus;
    align_sign = -1;
end

fprintf('Chosen alignment sign: %s (sum corr + = %.2f, - = %.2f)\n\n', ...
    ternary(align_sign==1,'PLUS','MINUS'), corr_plus, corr_minus);

% Align clean desired for evaluation reference
aligned_clean = zeros(Ns,N);
for k = 1:N
    aligned_clean(:,k) = interp1(t_vec, r_clean(:,k), t_vec + align_sign*delays_smooth(:,k), 'linear', 0);
end
s_ref_aligned = mean(aligned_clean,2);  % aligned desired reference

%% ===================== BASELINE (ONE AFFECTED MIC) =====================
eval_mic = affected_mics(round(end/2));      % e.g., mic 12
baseline_mic = aligned(:,eval_mic);

%% ===================== METHOD 1 (NON-ADAPTIVE) =====================
fprintf('=== Method 1: Non-adaptive reference subtraction + average ===\n');

clean_est = mean(aligned(:,unaffected_mics),2);

[b_lp,a_lp] = butter(4, 2000/(fs/2), 'low');

aligned_m1 = aligned;
for k = affected_mics
    i_est = aligned(:,k) - clean_est;
    i_est = filtfilt(b_lp,a_lp,i_est);
    aligned_m1(:,k) = aligned(:,k) - i_est;
end
s_hat_m1 = mean(aligned_m1,2);

%% ===================== METHOD 2 (ADAPTIVE NLMS ANC) =====================
% Uses a real interference reference mic from the affected side.
fprintf('=== Method 2: Adaptive ANC (NLMS) using interference reference mic ===\n');

aligned_m2 = aligned;

refIntMic = affected_mics(1);      % mic 9 as interference reference
x_ref_all = aligned(:, refIntMic);

Lw   = 256;     % filter length (64..512 typical)
mu   = 0.3;     % step size (0<mu<2, smaller is safer)
epsN = 1e-8;

% Log convergence on one representative affected mic
LOG_NLMS = true;
LOG_MICPICK = eval_mic;  % track same mic as baseline
nlms_e2 = []; nlms_wnorm = []; nlms_t = t_vec;

for k = affected_mics
    if k == refIntMic
        continue; 
    end

    d_sig = aligned(:,k);  % primary
    x_ref = x_ref_all;     % reference

    w = zeros(Lw,1);
    xbuf = zeros(Lw,1);
    e = zeros(Ns,1);

    doLog = LOG_NLMS && (k == LOG_MICPICK);
    if doLog
        nlms_e2 = zeros(Ns,1);
        nlms_wnorm = zeros(Ns,1);
    end

    for n = 1:Ns
        xbuf = [x_ref(n); xbuf(1:end-1)];
        yhat = w' * xbuf;
        e(n) = d_sig(n) - yhat;
        w = w + (mu/(xbuf'*xbuf + epsN)) * xbuf * e(n);

        if doLog
            nlms_e2(n)    = e(n)^2;
            nlms_wnorm(n) = norm(w,2);
        end
    end

    aligned_m2(:,k) = e;
end

s_hat_m2 = mean(aligned_m2,2);

%% ===================== METHOD 3 (BLUE/MVDR-like COMBINER) =====================
fprintf('=== Method 3: MVDR/BLUE-like after alignment (a_des = ones) ===\n');

a_des = ones(N,1);
sigma_sq = 1;
rho = 0.2;

Rn = zeros(N,N);
for i=1:N
    for j=1:N
        if i==j
            Rn(i,j) = sigma_sq;
        elseif abs(i-j)==1
            Rn(i,j) = rho*sigma_sq;
        else
            Rn(i,j) = 0;
        end
    end
end

q = Rn \ a_des;
w_blue = q / (a_des' * q);     % sum(w)=1 constraint
s_hat_m3 = aligned * w_blue;

%% ===================== EVALUATION (SIR-like + correlation) =====================
fprintf('\n=== Evaluation ===\n');

s_ref = s_ref_aligned(:);
s_ref_n = s_ref/(std(s_ref)+eps0);

yn  = @(x) x/(std(x)+eps0);
sir = @(y) 10*log10( var(s_ref_n(valid_idx)) / (var(y(valid_idx)-s_ref_n(valid_idx))+eps0) );
crr = @(y) abs(corr(y(valid_idx), s_ref_n(valid_idx)));

yB = yn(baseline_mic);
y1 = yn(s_hat_m1);
y2 = yn(s_hat_m2);
y3 = yn(s_hat_m3);

fprintf('Baseline (affected mic %d):   SIR=%.2f dB, Corr=%.3f\n', eval_mic, sir(yB), crr(yB));
fprintf('M1 Non-adaptive subtract:      SIR=%.2f dB, Corr=%.3f\n', sir(y1), crr(y1));
fprintf('M2 Adaptive NLMS ANC:          SIR=%.2f dB, Corr=%.3f\n', sir(y2), crr(y2));
fprintf('M3 BLUE/MVDR-like combiner:    SIR=%.2f dB, Corr=%.3f\n', sir(y3), crr(y3));

%% ===================== SAVE AUDIO FILES =====================
if SAVE_AUDIO_FILES
    fprintf('\nSaving audio files...\n');

    baseline_audio = baseline_mic/(max(abs(baseline_mic))+eps0);
    m1_audio = s_hat_m1/(max(abs(s_hat_m1))+eps0);
    m2_audio = s_hat_m2/(max(abs(s_hat_m2))+eps0);
    m3_audio = s_hat_m3/(max(abs(s_hat_m3))+eps0);
    ref_audio = s_ref/(max(abs(s_ref))+eps0);

    audiowrite('baseline_affected_mic.wav', baseline_audio, fs);
    audiowrite('m1_nonadaptive.wav',        [baseline_audio, m1_audio], fs);
    audiowrite('m2_nlms_anc.wav',           [baseline_audio, m2_audio], fs);
    audiowrite('m3_blue_mvdrlike.wav',      [baseline_audio, m3_audio], fs);
    audiowrite('desired_ref_aligned.wav',   ref_audio, fs);

    audiowrite('m1_nonadaptive_mono.wav', m1_audio, fs);
    audiowrite('m2_nlms_anc_mono.wav', m2_audio, fs);
    audiowrite('m3_blue_mvdrlike_mono.wav', m3_audio, fs);

    fprintf('Saved audio files.\n');
end

%% ===================== MSE =====================
% Normalize reference ONCE and gain-match outputs to reference (LS gain)
refStd = std(s_ref) + eps0;
s_ref_MSE = s_ref / refStd;

ls_gain = @(y, ref) ( (ref' * y) / (ref' * ref + eps0) );
scale_to_ref = @(y, ref) ls_gain(y, ref) * y;

yB_s = scale_to_ref(baseline_mic(:), s_ref);
y1_s = scale_to_ref(s_hat_m1(:),     s_ref);
y2_s = scale_to_ref(s_hat_m2(:),     s_ref);
y3_s = scale_to_ref(s_hat_m3(:),     s_ref);

yB_MSE = yB_s / refStd;
y1_MSE = y1_s / refStd;
y2_MSE = y2_s / refStd;
y3_MSE = y3_s / refStd;

mseB = mean( (yB_MSE(valid_idx) - s_ref_MSE(valid_idx)).^2 );
mse1 = mean( (y1_MSE(valid_idx) - s_ref_MSE(valid_idx)).^2 );
mse2 = mean( (y2_MSE(valid_idx) - s_ref_MSE(valid_idx)).^2 );
mse3 = mean( (y3_MSE(valid_idx) - s_ref_MSE(valid_idx)).^2 );

fprintf('\n=== MSE (valid region) [fixed: single normalization + LS gain match] ===\n');
fprintf('Baseline mic: %.6e\n', mseB);
fprintf('M1 Non-adapt: %.6e\n', mse1);
fprintf('M2 NLMS ANC : %.6e\n', mse2);
fprintf('M3 BLUE-like: %.6e\n', mse3);

figure('Name','Overall MSE comparison (fixed)','Position',[220 220 900 420]);
mse_all = [mseB mse1 mse2 mse3];
names = {'Baseline mic','M1 Non-adapt','M2 NLMS','M3 BLUE'};
bar(mse_all); grid on;
set(gca,'XTick',1:numel(names),'XTickLabel',names,'XTickLabelRotation',20);
ylabel('MSE vs aligned desired reference (gain-matched)');
title('Overall MSE (fixed)');

%% ===================== MSE vs THETA (per window, fixed scaling) =====================
fprintf('\nComputing MSE vs theta (per window)...\n');

mseB_w = zeros(1,num_w);
mse1_w = zeros(1,num_w);
mse2_w = zeros(1,num_w);
mse3_w = zeros(1,num_w);
theta_win = zeros(1,num_w);

for wi=1:num_w
    a = (wi-1)*hop + 1;
    b = a + win_len - 1;
    mid = round((a+b)/2);
    theta_win(wi) = theta_t(mid);

    mseB_w(wi)= mean((yB_MSE(a:b) - s_ref_MSE(a:b)).^2);
    mse1_w(wi)= mean((y1_MSE(a:b) - s_ref_MSE(a:b)).^2);
    mse2_w(wi)= mean((y2_MSE(a:b) - s_ref_MSE(a:b)).^2);
    mse3_w(wi)= mean((y3_MSE(a:b) - s_ref_MSE(a:b)).^2);
end

figure('Name','MSE vs Source Position (theta)','Position',[150 150 1100 480]);
plot(theta_win,mseB_w,'LineWidth',1.2); hold on;
plot(theta_win,mse1_w,'LineWidth',1.2);
plot(theta_win,mse2_w,'LineWidth',1.2);
plot(theta_win,mse3_w,'LineWidth',1.2);
grid on;
xlabel('\theta(t)=\omega_0 t (rad)');
ylabel('MSE vs aligned desired reference (gain-matched)');
title('MSE vs rotating source position');
legend('Baseline mic','M1 Non-adapt','M2 NLMS','M3 BLUE','Location','best');

%% ===================== EXTRA REPORT PLOTS =====================

% (1) Zoomed waveforms
tZoom1 = 2.00; tZoom2 = 2.05;
idxZ = (t_vec >= tZoom1 & t_vec <= tZoom2);

% Use the SAME normalization for visualization (reference std)
srefP = s_ref_MSE;     % already normalized by refStd
bP = yB_MSE; m1P = y1_MSE; m2P = y2_MSE; m3P = y3_MSE;

figure('Name','Waveforms (Zoom)','Position',[120 120 1100 420]);
plot(t_vec(idxZ), srefP(idxZ), 'k', 'LineWidth', 1.3); hold on;
plot(t_vec(idxZ), bP(idxZ),   'Color',[0.45 0.45 0.45], 'LineWidth', 1.0);
plot(t_vec(idxZ), m1P(idxZ),  'b--','LineWidth', 1.2);
plot(t_vec(idxZ), m2P(idxZ),  'g-.','LineWidth', 1.2);
plot(t_vec(idxZ), m3P(idxZ),  'r:','LineWidth', 1.4);
grid on; xlabel('Time (s)'); ylabel('Amplitude (ref-std normalized)');
title('Zoomed Waveforms: desired reference vs outputs');
legend('Aligned desired ref','Baseline mic','M1 non-adapt','M2 NLMS','M3 BLUE','Location','best');

% (2) Residual error (zoomed)
eB = bP - srefP;
e1 = m1P - srefP;
e2 = m2P - srefP;
e3 = m3P - srefP;

figure('Name','Residual Error (Zoom)','Position',[140 140 1100 420]);
plot(t_vec(idxZ), eB(idxZ), 'Color',[0.45 0.45 0.45], 'LineWidth', 1.0); hold on;
plot(t_vec(idxZ), e1(idxZ), 'b--','LineWidth', 1.2);
plot(t_vec(idxZ), e2(idxZ), 'g-.','LineWidth', 1.2);
plot(t_vec(idxZ), e3(idxZ), 'r:','LineWidth', 1.4);
grid on; xlabel('Time (s)'); ylabel('Residual');
title('Residual (Zoom): y(t) - aligned desired reference');
legend('Baseline mic','M1 non-adapt','M2 NLMS','M3 BLUE','Location','best');

% (3) Residual PSD (Welch)
nfft = 4096;
wlen = 2048;
over = 1024;
wW = hann(wlen);

eBv = eB(valid_idx); e1v = e1(valid_idx); e2v = e2(valid_idx); e3v = e3(valid_idx);

[PBe,f] = pwelch(eBv, wW, over, nfft, fs);
[P1e,~] = pwelch(e1v, wW, over, nfft, fs);
[P2e,~] = pwelch(e2v, wW, over, nfft, fs);
[P3e,~] = pwelch(e3v, wW, over, nfft, fs);

figure('Name','Residual PSD (Welch)','Position',[160 160 1100 420]);
plot(f, 10*log10(PBe+eps0), 'Color',[0.45 0.45 0.45], 'LineWidth', 1.0); hold on;
plot(f, 10*log10(P1e+eps0), 'b--','LineWidth', 1.2);
plot(f, 10*log10(P2e+eps0), 'g-.','LineWidth', 1.2);
plot(f, 10*log10(P3e+eps0), 'r:','LineWidth', 1.4);
grid on; xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
title('Residual PSD on valid region (lower is better)');
legend('Baseline mic','M1 non-adapt','M2 NLMS','M3 BLUE','Location','best');
xlim([0 8000]);

% (4) NLMS convergence
if ~isempty(nlms_e2) && ~isempty(nlms_wnorm)
    figure('Name','NLMS Convergence (Representative Mic)','Position',[180 180 1100 420]);
    yyaxis left;
    plot(nlms_t, 10*log10(nlms_e2+eps0), 'LineWidth', 1.2);
    ylabel('10log10(e^2) (dB)');

    yyaxis right;
    plot(nlms_t, nlms_wnorm, 'LineWidth', 1.2);
    ylabel('||w(n)||_2');

    grid on; xlabel('Time (s)');
    title(sprintf('NLMS Convergence (tracking mic %d)', LOG_MICPICK));
else
    warning('NLMS logging arrays are empty (check LOG_NLMS/LOG_MICPICK).');
end

fprintf('\n========== DONE ==========\n');

%% ===================== Local helper =====================
function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
