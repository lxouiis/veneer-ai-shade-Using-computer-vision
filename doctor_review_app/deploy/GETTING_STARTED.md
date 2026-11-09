# Getting Started: Zero to Deployed, for someone who's never touched AWS

This is the full path, start to finish, assuming you've never done any of
this before. Follow it in order - each part depends on the one before it.
`deploy/README.md` is the condensed reference version of parts of this;
this file is the "explain every click" version.

Rough time estimate: 2-4 hours spread across a few sessions, not something
to rush through in one sitting the first time.

---

## Part 0: What you need before starting
- A computer (you're on a Mac, so Terminal.app is your main tool here)
- A Google account (for Drive)
- An AWS account with some credit balance - you already have this (a new
  AWS account's standard signup credit, ~$200, not a student-specific
  program - a separate student credit application was submitted too but
  isn't required for any of this to work). **Skip straight to Part 2.**

## Part 1: AWS account + credits (reference only - you already have this)
For anyone else following this without an AWS account yet:
1. Go to `aws.amazon.com` and create an account - you'll need an email, a
   password, and a credit card (required even with credits, as a
   verification step - you won't be charged while credits cover usage).
2. New accounts typically get some starting credit automatically on
   signup. Student-specific programs (AWS Educate, GitHub Student Developer
   Pack) exist too and can add more, but approval can take time and isn't
   required to start - the standard signup credit is enough on its own.
3. Check your balance and expiration under **Billing and Cost Management ->
   Credits** in the AWS Console.

Also worth checking there: a brand-new account sometimes starts with low
default **service quotas** (e.g. how many vCPUs you can launch at once) -
if an instance type seems unavailable in Part 2, this is the likely reason,
not a credits problem. Picking a smaller instance (like `m7i-flex.large`
instead of `t3.medium`) usually sidesteps it without needing to request a
quota increase.

## Part 2: Launch the EC2 instance
This is the computer that will run everything - the app, the models.

1. Log into **console.aws.amazon.com**.
2. Top-right corner: pick a **region** close to you (e.g. `ap-south-1` for
   India). Write this down - you'll need it later for Part 10.
3. Search "EC2" in the top search bar, click into the EC2 service, click
   **Launch Instance**.
4. **Name**: anything, e.g. `tooth-review-app`.
5. **AMI (the operating system)**: search "Ubuntu", pick **Ubuntu Server
   22.04 LTS**.
6. **Instance type**: search for `t3.medium` first - if it's selectable,
   use it (cheapest option that's still big enough, ~$30/month if left
   running 24/7). If it doesn't show up (new accounts sometimes have a
   quota that excludes it), use **`m7i-flex.large`** instead - also fine,
   actually a bit more RAM, just pricier at ~$86/month if left running
   24/7. Either way, avoid anything with only 1-2 GiB memory (like
   `t3.micro`/`t3.small`) - not enough for the models. Stopping the
   instance between uses (Part 13) cuts this cost down a lot either way -
   you're only billed while it's actually running.
7. **Key pair**: click "Create new key pair". Name it, leave the defaults,
   click Create. **This downloads a `.pem` file to your Downloads folder -
   you cannot get this file again if you lose it, and you need it to
   connect to the instance. Move it somewhere safe now.**
8. **Network settings**: click "Edit". You'll see a rule for SSH (port 22) -
   change its source to **My IP** (so only you can SSH in). Click "Add
   security group rule" twice more: one for **HTTP** (port 80), source
   **Anywhere (0.0.0.0/0)**, one for **HTTPS** (port 443), source
   **Anywhere**. (These two let Doc Sans's browser reach the app later.)
9. **Storage**: change the default from 8 GiB to **20 GiB**.
10. Click **Launch Instance**. Wait a minute or two, then go to the EC2
    **Instances** page - it should show as "Running".

## Part 3: Give it a fixed address
By default, the instance's public IP changes if it ever restarts. Fix this:
1. EC2 sidebar -> **Elastic IPs** -> **Allocate Elastic IP address** ->
   Allocate.
2. Select the new address -> **Actions** -> **Associate Elastic IP address**
   -> pick your instance -> Associate.
3. Note this IP address down - you'll SSH into it and point your domain at it.

## Part 4: Connect to the instance
1. Open **Terminal** (Mac: Cmd+Space, type "Terminal").
2. Find your `.pem` file's location (likely `~/Downloads/your-key-name.pem`)
   and lock down its permissions (AWS requires this, it'll refuse to connect
   otherwise):
   ```bash
   chmod 400 ~/Downloads/your-key-name.pem
   ```
3. Connect (replace with your actual key path and Elastic IP):
   ```bash
   ssh -i ~/Downloads/your-key-name.pem ubuntu@<your-elastic-ip>
   ```
   First connection asks "are you sure you want to continue connecting?" -
   type `yes`. You're now inside the instance's terminal, not your Mac's.

## Part 5: Get a free domain name
HTTPS (the padlock) needs a real domain - a plain IP address can't get one.
1. Go to **duckdns.org**, sign in with Google/GitHub.
2. Type a subdomain name (e.g. `toothreview`), click **add domain**.
3. In the "current ip" field, paste your Elastic IP from Part 3, click
   **update ip**.
4. Your app's eventual address will be `https://toothreview.duckdns.org`
   (with whatever name you picked).

## Part 6: Set up a dedicated user and copy the app over
Back in your SSH terminal (still connected to the instance):
```bash
sudo adduser --disabled-password --gecos "" toothreview
sudo mkdir -p /home/toothreview/app
sudo chown -R toothreview:toothreview /home/toothreview/app
```

Now get the `doctor_review_app` folder onto the instance. Easiest way,
since this project isn't on GitHub yet: open a **second** Terminal tab on
your Mac (don't close the SSH one) and run, from your project folder:
```bash
scp -i ~/Downloads/your-key-name.pem -r doctor_review_app ubuntu@<your-elastic-ip>:/tmp/doctor_review_app
```
Back in your SSH terminal:
```bash
sudo mv /tmp/doctor_review_app/* /home/toothreview/app/
sudo chown -R toothreview:toothreview /home/toothreview/app
```

## Part 7: Install Python and the app's dependencies
```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv
sudo -u toothreview -i
cd ~/app
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p models
exit
```
(This takes a few minutes - `torch` is a large download.)

## Part 8: Get the two model checkpoint files
These are **not** in the GitHub-less copy you just scp'd over - they're
large binary files, published as Kaggle Datasets. Two files needed:
- `medsam_tooth_best.pth` - from the `notebook-a-meowmeow` Kaggle Dataset
- `tooth_yolo_best.pt` (the **original** detector, not the smile-finetuned
  one - see `notes.md` for why) - from the `yolomodel-inference` Kaggle
  Dataset, specifically `yolo_checkpoints/tooth_yolo/weights/best.pt`

On your Mac: go to kaggle.com, open each dataset, click **Download**. Then
upload both to the instance (second Terminal tab again):
```bash
scp -i ~/Downloads/your-key-name.pem medsam_tooth_best.pth ubuntu@<your-elastic-ip>:/tmp/
scp -i ~/Downloads/your-key-name.pem best.pt ubuntu@<your-elastic-ip>:/tmp/tooth_yolo_best.pt
```
Back in SSH:
```bash
sudo mv /tmp/medsam_tooth_best.pth /tmp/tooth_yolo_best.pt /home/toothreview/app/models/
sudo chown toothreview:toothreview /home/toothreview/app/models/*
```

## Part 9: The dataset - this goes to Google Drive, not the instance
The instance doesn't need the dataset directly - the app pulls images from
Drive at runtime. The dataset (NCT, and SmileImages if you want it in the
review queue too) needs downloading from Kaggle to **your Mac** first (same
`kaggle.com` download flow as Part 8), then pushed to Drive in Part 11.

## Part 10: Connect Google Drive (run this on your Mac, not the instance)
1. Go to **console.cloud.google.com**, create/select a project.
2. Search "Drive API" in the top search bar, click it, click **Enable**.
3. Left sidebar -> **Credentials** -> **Create Credentials** -> **OAuth
   client ID**. If asked, configure the consent screen first (External,
   fill in an app name, your email - keep it in "Testing" mode, that's
   fine for this). For the client ID itself, **Application type: Desktop
   app**, create it, then **Download JSON**.
4. Rename the downloaded file to `client_secret.json`, place it inside your
   local `doctor_review_app/` folder (next to `setup_drive.py`).
5. In Terminal, on your Mac:
   ```bash
   cd doctor_review_app
   pip install -r requirements.txt
   python setup_drive.py
   ```
   A browser window opens - sign in with the Google account that should own
   this data, click Allow. Back in Terminal, it prints four values:
   `GDRIVE_CLIENT_ID`, `GDRIVE_CLIENT_SECRET`, `GDRIVE_REFRESH_TOKEN`,
   `GDRIVE_ROOT_ID`. **Copy all four somewhere safe** - you'll need them in
   Part 12.

## Part 11: Upload the dataset to Drive (also run on your Mac)
```bash
python upload_dataset_to_drive.py /path/to/the/dataset/images/you/downloaded
```
This can take a while for hundreds of images - it prints progress and is
safe to re-run if interrupted.

## Part 12: Put the secrets on the instance
Back in your SSH terminal:
```bash
sudo -u toothreview tee /home/toothreview/app/.env > /dev/null <<'EOF'
GDRIVE_CLIENT_ID=paste_value_from_part_10
GDRIVE_CLIENT_SECRET=paste_value_from_part_10
GDRIVE_REFRESH_TOKEN=paste_value_from_part_10
GDRIVE_ROOT_ID=paste_value_from_part_10
MEDSAM_CHECKPOINT_PATH=/home/toothreview/app/models/medsam_tooth_best.pth
YOLO_CHECKPOINT_PATH=/home/toothreview/app/models/tooth_yolo_best.pt
EOF
sudo chmod 600 /home/toothreview/app/.env
```
(One more line, `LAMBDA_STARTER_URL`, gets added here too - but not until
Part 14, since the Lambda doesn't exist yet. Don't skip that follow-up step
later, or the app's "Stop Instance" button in the sidebar just won't appear.)

## Part 13: Make the app run automatically, and give it HTTPS
```bash
sudo cp /home/toothreview/app/deploy/tooth-review.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tooth-review
sudo systemctl status tooth-review   # should say "active (running)" - Ctrl+C to exit the status view
```
Then Caddy (gives you the real `https://` link):
```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy
```
Edit the domain inside the Caddy config before copying it in:
```bash
sudo nano /home/toothreview/app/deploy/Caddyfile
```
Replace `your-subdomain.duckdns.org` with your actual DuckDNS domain from
Part 5 (arrow keys to move, Ctrl+O then Enter to save, Ctrl+X to exit), then:
```bash
sudo cp /home/toothreview/app/deploy/Caddyfile /etc/caddy/Caddyfile
sudo systemctl restart caddy
```
**Open `https://yourname.duckdns.org` in a browser now - this should show
the app.** If it doesn't load yet, give it a minute - the first TLS
certificate issuance takes a moment.

## Part 14: The one-tap start/stop button (so the instance can stay off between uses)
1. AWS Console -> search "Lambda" -> **Create function**.
2. Name it (e.g. `tooth-review-starter`), Runtime: **Python 3.12**, Create.
3. In the code editor that opens, delete the placeholder code and paste in
   the contents of `deploy/lambda_starter/start_stop_ec2.py`. Click **Deploy**.
4. **Configuration** tab -> **Environment variables** -> Add: key
   `INSTANCE_ID`, value = your EC2 instance's ID (find it on the EC2
   Instances page, looks like `i-0abc123...`).
5. **Configuration** tab -> **Permissions** -> click the role name link -> 
   **Add permissions** -> **Create inline policy** -> switch to the **JSON**
   tab -> paste in `deploy/lambda_starter/iam_policy.json`, but first fill
   in your actual region (from Part 2), account ID (top-right of the AWS
   console, under your name), and instance ID. Save.
6. **Configuration** tab -> **Function URL** -> **Create function URL** ->
   Auth type: **NONE** -> Save. **Copy this URL.**
7. Back in your SSH terminal, add this URL to the app's `.env` so the
   "Stop Instance" button shows up in the app's sidebar:
   ```bash
   echo "LAMBDA_STARTER_URL=paste_the_function_url_here" | sudo -u toothreview tee -a /home/toothreview/app/.env
   sudo systemctl restart tooth-review
   ```

## Part 15: The bookmarkable start page
1. AWS Console -> search "S3" -> **Create bucket**. Give it a unique name,
   leave other settings default for now, Create.
2. Open the bucket -> **Properties** tab -> scroll to **Static website
   hosting** -> Edit -> Enable -> Index document: `start_page.html` -> Save.
3. **Permissions** tab -> **Block public access** -> Edit -> uncheck "Block
   all public access" -> confirm. Then **Bucket policy** -> Edit -> paste:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Principal": "*",
       "Action": "s3:GetObject",
       "Resource": "arn:aws:s3:::YOUR-BUCKET-NAME/*"
     }]
   }
   ```
   (replace `YOUR-BUCKET-NAME`), Save.
4. On your Mac, open `deploy/lambda_starter/start_page.html` in a text
   editor. Replace `REPLACE_WITH_YOUR_LAMBDA_FUNCTION_URL` with the URL from
   Part 14 step 6, and `REPLACE_WITH_YOUR_DUCKDNS_DOMAIN` with your
   `https://yourname.duckdns.org` link from Part 5.
5. Back in the S3 bucket -> **Objects** tab -> **Upload** -> select the
   edited `start_page.html` -> Upload.
6. **Properties** tab again -> scroll to **Static website hosting** -> copy
   the **Bucket website endpoint** URL shown there. **This is the link Doc
   Sans bookmarks on her iPad** - not the DuckDNS one.

## Part 16: Test the whole flow
1. Stop the instance (EC2 Instances page -> select it -> Instance state ->
   Stop), to simulate "nothing running."
2. Open the S3 website endpoint link from Part 15 in a browser, click
   **Start**.
3. It should say "Starting instance... Booting up..." and within about a
   minute, auto-redirect to the real app, now showing a photo and predicted
   mask.

If every step above worked, this is done - that S3 link is the one and only
thing Doc Sans needs saved anywhere.

## If something breaks
- `sudo systemctl status tooth-review` - shows the app's own error output if
  it's not running.
- `sudo journalctl -u tooth-review -n 50` - more detailed recent logs.
- `sudo systemctl status caddy` - same, for the HTTPS layer.
- Come back here with the exact error text, not just "it didn't work."
